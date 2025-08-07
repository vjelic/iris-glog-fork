import logging
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the iris package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Test logging functionality without requiring full iris environment
def test_logging_constants():
    """Test that logging constants are properly defined."""
    # Import iris.iris module directly to avoid MPI dependencies
    from iris.iris import DEBUG, INFO, WARNING, ERROR

    # Verify constants match Python logging levels
    assert DEBUG == logging.DEBUG
    assert INFO == logging.INFO
    assert WARNING == logging.WARNING
    assert ERROR == logging.ERROR


def test_set_logger_level():
    """Test the set_logger_level function."""
    from iris.iris import set_logger_level, logger, DEBUG, INFO

    # Test setting different levels
    set_logger_level(DEBUG)
    assert logger.level == logging.DEBUG

    set_logger_level(INFO)
    assert logger.level == logging.INFO


def test_logger_setup():
    """Test that the iris logger is properly configured."""
    from iris.iris import logger

    # Verify logger name
    assert logger.name == "iris"

    # Verify default level
    assert logger.level == logging.INFO

    # Verify handler exists
    assert len(logger.handlers) > 0

    # Verify handler is a StreamHandler
    assert isinstance(logger.handlers[0], logging.StreamHandler)


@patch("iris.iris.logger")
def test_iris_debug_logging(mock_logger):
    """Test that Iris debug logging uses Python logger directly."""
    # Mock the MPI and other dependencies to create a minimal Iris instance
    with patch("iris.iris.init_mpi", return_value=(None, 0, 1)):
        with patch("iris.iris.count_devices", return_value=1):
            with patch("iris.iris.set_device"):
                with patch("iris.iris.torch.empty"):
                    with patch("iris.iris.get_ipc_handle"):
                        with patch("iris.iris.world_barrier"):
                            with patch("iris.iris.mpi_allgather", return_value=[]):
                                with patch("iris.iris.open_ipc_handle"):
                                    with patch("iris.iris.torch.from_numpy"):
                                        from iris.iris import Iris

                                        # Create a minimal Iris instance
                                        iris_instance = Iris.__new__(Iris)
                                        iris_instance.cur_rank = 0
                                        iris_instance.num_ranks = 1

                                        # Test allocate method debug logging
                                        iris_instance.allocate(100, None)
                                        mock_logger.debug.assert_called_with(
                                            "[0/1] allocate: num_elements = 100, dtype = None"
                                        )


def test_logger_api_usage():
    """Test direct logger API usage."""
    from iris.iris import logger, set_logger_level, DEBUG, INFO

    # Capture log output
    import io
    import logging

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.Formatter("[Iris] %(message)s"))

    # Remove existing handlers and add our capture handler
    logger.handlers.clear()
    logger.addHandler(handler)

    # Test logging at different levels
    set_logger_level(INFO)
    logger.info("Test info message")
    logger.debug("Test debug message (should be hidden)")

    set_logger_level(DEBUG)
    logger.debug("Test debug message (should be visible)")

    output = log_capture.getvalue()
    assert "[Iris] Test info message" in output
    assert "[Iris] Test debug message (should be visible)" in output
    # The hidden debug message should not appear
    lines = output.split("\n")
    hidden_debug_count = sum(1 for line in lines if "should be hidden" in line)
    assert hidden_debug_count == 0


def test_api_import():
    """Test that the new API can be imported from the main iris module."""
    # This test verifies the __init__.py exports work correctly
    try:
        from iris import set_logger_level, logger, DEBUG, INFO, WARNING, ERROR

        # If we get here, the imports worked
        assert set_logger_level is not None
        assert logger is not None
        assert logger.name == "iris"
        assert DEBUG == logging.DEBUG
        assert INFO == logging.INFO
        assert WARNING == logging.WARNING
        assert ERROR == logging.ERROR
    except ImportError as e:
        # If iris module can't be imported due to dependencies, skip this test
        pytest.skip(f"Skipping API import test due to dependency issues: {e}")


if __name__ == "__main__":
    # Run basic tests
    test_logging_constants()
    test_set_logger_level()
    test_logger_setup()
    test_logger_api_usage()
    print("All basic logging tests passed!")
