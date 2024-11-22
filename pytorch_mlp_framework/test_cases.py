import torch
from model_architectures import (
    ConvolutionalProcessingBlock,
    ProcessingBlockBN,
    ProcessingBlockBNRC,
    DownsamplingBlockBN,
    ConvolutionalDimensionalityReductionBlock,
)

def test_convolutional_processing_block():
    print("Testing ConvolutionalProcessingBlock...")
    input_tensor = torch.randn(8, 3, 32, 32)  # Batch size: 8, Channels: 3, Height: 32, Width: 32
    block = ConvolutionalProcessingBlock(input_shape=(8, 3, 32, 32), num_filters=16, kernel_size=3, padding=1, bias=False, dilation=1)
    output = block(input_tensor)
    assert output.shape == (8, 16, 32, 32), f"Expected shape (8, 16, 32, 32), got {output.shape}"
    print("ConvolutionalProcessingBlock passed!")

def test_processing_block_bn():
    print("Testing ProcessingBlockBN...")
    input_tensor = torch.randn(8, 3, 32, 32)
    block = ProcessingBlockBN(input_shape=(8, 3, 32, 32), num_filters=16, kernel_size=3, padding=1, bias=False, dilation=1)
    output = block(input_tensor)
    assert output.shape == (8, 16, 32, 32), f"Expected shape (8, 16, 32, 32), got {output.shape}"
    print("ProcessingBlockBN passed!")

def test_processing_block_bnrc():
    print("Testing ProcessingBlockBNRC...")
    input_tensor = torch.randn(8, 3, 32, 32)
    block = ProcessingBlockBNRC(input_shape=(8, 3, 32, 32), num_filters=3, kernel_size=3, padding=1, bias=False, dilation=1)
    output = block(input_tensor)
    assert output.shape == (8, 3, 32, 32), f"Expected shape (8, 3, 32, 32), got {output.shape}"
    print("ProcessingBlockBNRC passed!")

def test_downsampling_block_bn():
    print("Testing DownsamplingBlockBN...")
    input_tensor = torch.randn(8, 3, 32, 32)
    block = DownsamplingBlockBN(input_shape=(8, 3, 32, 32), num_filters=16, kernel_size=3, padding=1, bias=False, dilation=1, reduction_factor=2)
    output = block(input_tensor)
    assert output.shape == (8, 16, 16, 16), f"Expected shape (8, 16, 16, 16), got {output.shape}"
    print("DownsamplingBlockBN passed!")

def test_dimensionality_reduction_block():
    print("Testing ConvolutionalDimensionalityReductionBlock...")
    input_tensor = torch.randn(8, 3, 32, 32)
    block = ConvolutionalDimensionalityReductionBlock(input_shape=(8, 3, 32, 32), num_filters=16, kernel_size=3, padding=1, bias=False, dilation=1, reduction_factor=2)
    output = block(input_tensor)
    assert output.shape == (8, 16, 16, 16), f"Expected shape (8, 16, 16, 16), got {output.shape}"
    print("ConvolutionalDimensionalityReductionBlock passed!")

def run_all_tests():
    print("Running all tests...")
    test_convolutional_processing_block()
    test_processing_block_bn()
    test_processing_block_bnrc()
    test_downsampling_block_bn()
    test_dimensionality_reduction_block()
    print("All tests passed!")

if __name__ == "__main__":
    run_all_tests()

