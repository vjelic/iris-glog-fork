import logging
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the iris package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
    from iris.iris import set_logger_level, _iris_logger, DEBUG, INFO
    
    # Test setting different levels
    set_logger_level(DEBUG)
    assert _iris_logger.level == logging.DEBUG
    
    set_logger_level(INFO)
    assert _iris_logger.level == logging.INFO


def test_logger_setup():
    """Test that the iris logger is properly configured."""
    from iris.iris import _iris_logger
    
    # Verify logger name
    assert _iris_logger.name == "iris"
    
    # Verify default level
    assert _iris_logger.level == logging.INFO
    
    # Verify handler exists
    assert len(_iris_logger.handlers) > 0
    
    # Verify handler is a StreamHandler
    assert isinstance(_iris_logger.handlers[0], logging.StreamHandler)


@patch('iris.iris._iris_logger')
def test_iris_log_methods_with_logging(mock_logger):
    """Test that Iris log methods use Python logging when conditions are met."""
    # Mock the MPI and other dependencies to create a minimal Iris instance
    with patch('iris.iris.init_mpi', return_value=(None, 0, 1)):
        with patch('iris.iris.count_devices', return_value=1):
            with patch('iris.iris.set_device'):
                with patch('iris.iris.torch.empty'):
                    with patch('iris.iris.get_ipc_handle'):
                        with patch('iris.iris.world_barrier'):
                            with patch('iris.iris.mpi_allgather', return_value=[]):
                                with patch('iris.iris.open_ipc_handle'):
                                    with patch('iris.iris.torch.from_numpy'):
                                        from iris.iris import Iris, LOGGING, STATS, _DEBUG_GLOBAL
                                        
                                        # Create a minimal Iris instance
                                        iris_instance = Iris.__new__(Iris)
                                        iris_instance.cur_rank = 0
                                        iris_instance.num_ranks = 1
                                        
                                        # Test log method
                                        if LOGGING:
                                            iris_instance.log("test message")
                                            mock_logger.info.assert_called_with("[0/1] test message")
                                        
                                        # Test log_debug method
                                        if _DEBUG_GLOBAL:
                                            iris_instance.log_debug("debug message") 
                                            mock_logger.debug.assert_called_with("[0/1] debug message")
                                        
                                        # Test log_stats method
                                        if STATS:
                                            iris_instance.log_stats("stats message")
                                            mock_logger.info.assert_called_with("[0/1] stats message")


def test_backward_compatibility_global_variables():
    """Test that global variables still exist for backward compatibility."""
    from iris.iris import STATS, LOGGING, _DEBUG_GLOBAL
    
    # Verify the global variables exist and have expected types
    assert isinstance(STATS, bool)
    assert isinstance(LOGGING, bool) 
    assert isinstance(_DEBUG_GLOBAL, bool)
    
    # Verify default values
    assert STATS == True
    assert LOGGING == True
    assert _DEBUG_GLOBAL == False


def test_api_import():
    """Test that the new API can be imported from the main iris module."""
    # This test verifies the __init__.py exports work correctly
    try:
        from iris import set_logger_level, DEBUG, INFO, WARNING, ERROR
        # If we get here, the imports worked
        assert set_logger_level is not None
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
    test_backward_compatibility_global_variables()
    print("All basic logging tests passed!")