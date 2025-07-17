# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import gzip
import json
import logging
import os
import re
import shutil
import string
import tempfile
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

def perf_func(func, iters, warmup_iters):
    """
    Measures the average execution time of a function on the GPU.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    output = None
    
    # Warmup iterations
    for _ in range(warmup_iters):
        output = func()
    
    # Timed iterations
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        output = func()
    stop_event.record()
    torch.cuda.synchronize()
    
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def dist_print(*args, **kwargs):
    """
    Prints from specified ranks to avoid cluttered output in distributed environments.
    """
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    allowed_ranks = kwargs.pop("allowed_ranks", [0])
    need_sync = kwargs.pop("need_sync", False)

    if isinstance(allowed_ranks, str) and allowed_ranks.lower() == "all":
        allowed_ranks = list(range(world_size))

    if need_sync:
        torch.distributed.barrier()

    if rank in allowed_ranks:
        print(*args, **kwargs)

###
# Profiler Utilities
###

def load_json(json_file):
    with open(json_file, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()
        content = content.encode().decode("unicode_escape")

        def replace_non_ascii_and_quotes(match):
            name = match.group(1)
            visible_printable = "".join(c for c in string.printable if c not in "\t\n\r\x0b\x0c}{")
            cleaned_name = "".join(c if c in visible_printable else "x" for c in name)
            cleaned_name = cleaned_name.replace('"', "y")
            return f'"name": "{cleaned_name}"'

        cleaned_content = re.sub(
            r'"name": "([\s\S]*?)"(?=, |\}|\s*\})',
            replace_non_ascii_and_quotes,
            content,
            flags=re.DOTALL,
        )
    return json.loads(cleaned_content, strict=False)


def process_trace_json(json_file):
    RANK_MAX_PID = 100000000

    def _mapping(x, delta):
        return f"{x}_{delta}" if isinstance(x, str) else x + delta

    def _process_item(item, rank, delta):
        item["pid"] = _mapping(item["pid"], delta)
        item["tid"] = _mapping(item["tid"], delta)
        if item["ph"] == "M":
            if item["name"] in ["process_name", "thread_name"]:
                item["args"]["name"] = f"{item['args']['name']}_rank{rank}"
            elif item["name"] == "process_labels":
                item["args"]["labels"] = f"{item['args']['labels']}_{rank}"

    trace = load_json(json_file)
    events = trace.get("traceEvents", [])
    rank = trace.get("distributedInfo", {}).get("rank", 0)
    delta = rank * RANK_MAX_PID
    for x in events:
        _process_item(x, rank, delta)
    return trace


class ParallelJsonDumper:
    def __init__(self, parallel_field: str, chunk_size: int = 5000):
        self.chunk_size = chunk_size
        self.cpu_count = cpu_count()
        self.parallel_field = parallel_field

    def dump(self, data: Dict[str, Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pvalue = data.pop(self.parallel_field, [])

        chunks = [pvalue[i:i + self.chunk_size] for i in range(0, len(pvalue), self.chunk_size)]

        with Pool(processes=min(len(chunks), self.cpu_count)) as pool:
            chunk_strings = pool.map(self._process_chunk, chunks)
            self._write_output(data, chunk_strings, output_path)

    def _process_chunk(self, chunk: List[Any]) -> str:
        chunk_json = json.dumps(chunk, separators=(",", ":"))
        return chunk_json[1:-1]

    def _write_output(self, base_data: Dict[str, Any], chunk_strings: List[str], output_path: Path) -> None:
        with open(output_path, "w") as f:
            f.write(json.dumps(base_data, separators=(",", ":"))[:-1])
            f.write(f',"{self.parallel_field}":[')
            for i, chunk_str in enumerate(chunk_strings):
                if i > 0 and chunk_str:
                    f.write(",")
                f.write(chunk_str)
            f.write("]}")


def _merge_json(to_merge_files: List[Path], output_json: Path, compress: bool = True):
    events = []
    trace = {}
    with Pool(processes=min(len(to_merge_files), cpu_count())) as pool:
        results = pool.map(process_trace_json, to_merge_files)
        for r in results:
            events.extend(r.get("traceEvents", []))
        if results:
            trace = results[-1] # Use the last trace as a template

    trace["traceEvents"] = events
    logging.info("Dumping merged json profile...")
    ParallelJsonDumper("traceEvents", 100000).dump(trace, Path(output_json))

    if compress:
        logging.info("Compressing profile...")
        with open(output_json, "rb") as f_in, gzip.open(output_json.with_suffix(".json.gz"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        output_json.unlink()
    logging.info("Done.")


class group_profile:
    def __init__(self, name: str, do_prof: bool, group: Optional[torch.distributed.ProcessGroup], **kwargs):
        self.name = name
        self.do_prof = do_prof
        self.group = group or torch.distributed.group.WORLD
        self.merge_group = kwargs.get("merge_group", True)
        self.keep_merged_only = kwargs.get("keep_merged_only", True)
        self.compress = kwargs.get("compress", True)
        
        self.profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) if self.do_prof else contextmanager(lambda: iter([None]))()

        self.trace_file = Path("prof") / f"{self.name}" / f"rank{self.group.rank()}.json"

    def __enter__(self):
        self.profile.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profile.__exit__(exc_type, exc_val, exc_tb)
        if self.do_prof:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Exporting chrome trace to {self.trace_file}")
            self.profile.export_chrome_trace(str(self.trace_file))
            if self.merge_group:
                self.merge_all()

    def merge_all(self):
        self.group.barrier()
        trace_content_list = [None] * self.group.size()
        if self.group.rank() == 0:
            for i in range(self.group.size()):
                trace_path = Path("prof") / f"{self.name}" / f"rank{i}.json"
                if trace_path.exists():
                    with open(trace_path, "rb") as f:
                        trace_content_list[i] = f.read()

        torch.distributed.broadcast_object_list([trace_content_list], src=0, group=self.group)

        if self.group.rank() == 0:
            logging.info("Merging profiles...")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                to_merge_files = []
                for i, content in enumerate(trace_content_list):
                    if content:
                        trace_file = tmp_path / f"trace_{i}.json"
                        with open(trace_file, "wb") as f:
                            f.write(content)
                        to_merge_files.append(trace_file)

                if to_merge_files:
                    merged_json = Path("prof") / f"{self.name}_merged.json"
                    _merge_json(to_merge_files, merged_json, self.compress)

        self.group.barrier()
        if self.keep_merged_only:
            shutil.rmtree(self.trace_file.parent, ignore_errors=True)