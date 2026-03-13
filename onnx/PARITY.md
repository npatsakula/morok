# ONNX Operator Parity

**160 / 198** standard operators implemented (80%).

Test results from the ONNX backend node test suite: **1361** pass, **0** fail, **317** skip (out of 1678 tests).

The *expanded uses* column counts how many `_expanded` tests exercise each
operator as a building block (indirect coverage beyond direct tests).

## Arithmetic

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Abs | Y | 1 pass | - | 9 |
| Add | Y | 8 pass | - | 127 |
| Div | Y | 10 pass | - | 145 |
| Mean | Y | 3 pass | - | - |
| Mod | Y | 13 pass | - | 62 |
| Mul | Y | 9 pass | - | 163 |
| Neg | Y | 2 pass | - | 28 |
| Pow | Y | 12 pass | - | 3 |
| Sub | Y | 9 pass | 7 pass | 69 |
| Sum | Y | 3 pass | - | 4 |

## Bitwise

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| BitShift | Y | 8 pass | - | - |
| BitwiseAnd | Y | 4 pass | - | - |
| BitwiseNot | Y | 3 pass | - | - |
| BitwiseOr | Y | 4 pass | - | - |
| BitwiseXor | Y | 4 pass | - | - |

## Math

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Acos | Y | 2 pass | - | - |
| Acosh | Y | 2 pass | - | - |
| Asin | Y | 2 pass | - | - |
| Asinh | Y | 2 pass | - | - |
| Atan | Y | 2 pass | - | - |
| Atanh | Y | 2 pass | - | - |
| Ceil | Y | 2 pass | - | 2 |
| Cos | Y | 2 pass | - | 6 |
| Cosh | Y | 2 pass | - | - |
| Erf | Y | 1 pass | - | 2 |
| Exp | Y | 2 pass | 2 pass | 23 |
| Floor | Y | 2 pass | - | - |
| IsInf | Y | 4 pass | - | - |
| IsNaN | Y | 2 pass | - | - |
| Log | Y | 2 pass | - | 21 |
| Reciprocal | Y | 2 pass | - | 19 |
| Round | Y | 1 pass | - | 3 |
| Sign | Y | 1 pass | - | - |
| Sin | Y | 2 pass | - | - |
| Sinh | Y | 2 pass | - | - |
| Sqrt | Y | 2 pass | - | 116 |
| Tan | Y | 2 pass | - | - |
| Det | Y | 2 pass | - | - |

## Activation

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Celu | Y | 1 pass | 1 pass | - |
| Elu | Y | 3 pass | - | 1 |
| Gelu | Y | 4 pass | 4 pass | - |
| HardSigmoid | Y | 3 pass | - | 1 |
| HardSwish | Y | 1 pass | 1 pass | - |
| LeakyRelu | Y | 3 pass | 3 pass | - |
| Mish | Y | 1 pass | 1 pass | - |
| PRelu | Y | 2 pass | 2 pass | - |
| Relu | Y | 1 pass | - | 2 |
| Selu | Y | 3 pass | - | - |
| Sigmoid | Y | 2 pass | - | 1 |
| Softmax | Y | 7 pass | 7 pass | 62 |
| LogSoftmax | Y | 7 pass | 7 pass | 34 |
| Softplus | Y | 2 pass | - | 1 |
| Softsign | Y | 2 pass | - | - |
| Swish | Y | 1 pass | 1 pass | - |
| Tanh | Y | 2 pass | - | 11 |
| ThresholdedRelu | Y | 3 pass | - | - |

## Comparison & Logic

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| And | Y | 8 pass | - | 62 |
| Equal | Y | 8 pass, 2 skip | - | 89 |
| Greater | Y | 8 pass | - | 11 |
| GreaterOrEqual | Y | 8 pass | 8 pass | - |
| Less | Y | 8 pass | - | 34 |
| LessOrEqual | Y | 8 pass | 8 pass | - |
| Not | Y | 3 pass | - | 62 |
| Or | Y | 8 pass | - | 16 |
| Where | Y | 2 pass | - | 84 |
| Xor | Y | 8 pass | - | - |

## Conditional & Selection

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Clip | Y | 12 pass | 12 pass | 3 |
| Hardmax | Y | 7 pass | - | - |
| Max | Y | 14 pass | - | 9 |
| Min | Y | 14 pass | - | 3 |
| Shrink | Y | 2 pass | - | - |

## Type & Cast

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Cast | Y | 16 pass, 44 skip | 19 pass | 205 |
| CastLike | Y | 16 pass, 40 skip | 33 pass, 40 skip | 33 |
| DequantizeLinear | - | 14 skip | - | - |
| DynamicQuantizeLinear | - | 3 skip | 3 skip | - |
| QLinearConv | Y | 1 pass | - | - |
| QLinearMatMul | Y | 8 pass | - | - |
| QuantizeLinear | - | 13 skip | - | 3 |

## Shape & Transform

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| CenterCropPad | Y | 6 pass | 6 pass | - |
| Concat | Y | 12 pass | - | 101 |
| ConstantOfShape | Y | 2 pass, 1 skip | - | 50 |
| DepthToSpace | Y | 2 pass | - | - |
| Expand | Y | 2 pass | - | 62 |
| EyeLike | Y | 3 pass | - | - |
| Flatten | Y | 9 pass | - | 19 |
| Pad | Y | 6 pass | - | 7 |
| Range | Y | 2 pass | 2 skip | 40 |
| Reshape | Y | 10 pass | - | 129 |
| Shape | Y | 11 pass | - | 150 |
| Size | Y | 2 pass | - | 42 |
| Slice | Y | 8 pass | - | 55 |
| SpaceToDepth | Y | 2 pass | - | - |
| Split | Y | 16 pass | - | 8 |
| Squeeze | Y | 2 pass | - | 29 |
| Tile | Y | 2 pass | - | - |
| Transpose | Y | 7 pass | - | 107 |
| Unsqueeze | Y | 7 pass | - | 92 |
| Col2Im | Y | 5 pass | - | - |
| ReverseSequence | Y | 2 pass | - | - |
| SplitToSequence | - | 3 skip | - | - |
| Unique | - | 6 skip | - | - |

## Indexing & Gather

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Compress | Y | 4 pass | - | - |
| CumSum | Y | 9 pass | - | - |
| Gather | Y | 4 pass | - | 27 |
| GatherElements | Y | 3 pass | - | 18 |
| GatherND | Y | 3 pass | - | - |
| NonZero | Y | 1 pass | - | - |
| OneHot | Y | 4 pass | - | - |
| Scatter | Y | 2 pass | - | - |
| ScatterElements | Y | 6 pass | - | - |
| ScatterND | Y | 5 pass | - | - |
| TopK | Y | 7 pass | - | - |
| Trilu | Y | 18 pass | - | - |
| TensorScatter | Y | 3 pass | - | - |

## Reduction

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| ArgMax | Y | 16 pass | - | - |
| ArgMin | Y | 16 pass | - | - |
| ReduceL1 | Y | 9 pass | 9 pass | - |
| ReduceL2 | Y | 9 pass | 9 pass | - |
| ReduceLogSum | Y | 5 pass | 5 pass | - |
| ReduceLogSumExp | Y | 9 pass | 9 pass | - |
| ReduceMax | Y | 10 pass | 7 pass | 17 |
| ReduceMean | Y | 8 pass | 1 pass | 43 |
| ReduceMin | Y | 10 pass | - | 3 |
| ReduceProd | Y | 9 pass | - | - |
| ReduceSum | Y | 12 pass | - | 66 |
| ReduceSumSquare | Y | 9 pass | 9 pass | - |

## Neural Network

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| AffineGrid | Y | 4 pass | 4 skip | - |
| Attention | Y | 62 pass | 62 pass | - |
| AveragePool | Y | 20 pass | - | - |
| BatchNormalization | Y | 2 pass, 2 skip | - | - |
| Conv | Y | 6 pass | - | - |
| ConvTranspose | Y | 11 pass | - | - |
| Dropout | Y | 5 pass, 7 skip | - | - |
| Einsum | Y | 6 pass | - | - |
| Gemm | Y | 11 pass | - | - |
| GlobalAveragePool | Y | 2 pass | - | - |
| GlobalMaxPool | Y | 2 pass | - | - |
| GridSample | Y | 18 pass | - | - |
| GroupNormalization | Y | 2 pass | 2 pass | - |
| InstanceNormalization | Y | 2 pass | - | - |
| LRN | Y | 2 pass | - | - |
| LayerNormalization | Y | 19 pass | 19 pass | - |
| LpNormalization | Y | 6 pass | - | - |
| MatMul | Y | 7 pass | - | 66 |
| MatMulInteger | Y | 1 pass | - | - |
| MaxPool | Y | 19 pass | - | - |
| MeanVarianceNormalization | Y | 1 pass | 1 pass | - |
| NegativeLogLikelihoodLoss | Y | 18 pass | 18 pass | 34 |
| RMSNormalization | Y | 19 pass | 19 pass | - |
| RNN | Y | 4 pass | - | - |
| Resize | Y | 39 pass | - | - |
| RotaryEmbedding | Y | 8 pass | 8 pass | - |
| SoftmaxCrossEntropyLoss | Y | 34 pass | 34 pass | - |
| Upsample | Y | 1 pass | - | - |
| ConvInteger | Y | 2 pass | - | - |
| DeformConv | - | 4 skip | - | - |
| GRU | Y | 4 pass | - | - |
| GlobalLpPool | Y | - | - | - |
| LSTM | Y | 4 pass | - | - |
| LpPool | Y | 8 pass | - | - |
| MaxRoiPool | - | - | - | - |
| MaxUnpool | Y | 2 pass | - | - |
| NonMaxSuppression | - | 10 skip | - | - |
| RoiAlign | - | 3 skip | - | - |

## Control Flow

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| If | Y | 1 pass, 2 skip | - | 4 |
| Loop | - | 3 skip | - | 8 |
| Scan | - | 2 skip | - | - |

## Constants & Identity

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Constant | Y | 1 pass | - | 213 |
| Identity | Y | 1 pass, 2 skip | - | 99 |
| OptionalGetElement | Y | 4 skip | - | - |
| OptionalHasElement | Y | 7 skip | - | - |
| Optional | - | - | - | - |

## Sequence

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| ConcatFromSequence | - | - | - | - |
| SequenceAt | - | - | - | - |
| SequenceConstruct | - | - | - | - |
| SequenceEmpty | - | - | - | 6 |
| SequenceErase | - | - | - | - |
| SequenceInsert | - | 2 skip | - | - |
| SequenceLength | - | - | - | 6 |
| SequenceMap | - | 6 skip | 6 skip | - |

## Random

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| Bernoulli | - | 3 skip | 3 skip | - |
| Multinomial | - | - | - | - |
| RandomNormal | - | - | - | - |
| RandomNormalLike | - | - | - | - |
| RandomUniform | - | - | - | - |
| RandomUniformLike | - | - | - | 3 |

## Signal & Text

| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |
|----------|------|-------------------|-------------------|---------------|
| BlackmanWindow | - | 2 skip | 2 skip | - |
| DFT | - | 10 skip | - | - |
| HammingWindow | - | 2 skip | 2 skip | - |
| HannWindow | - | 2 skip | 2 skip | - |
| ImageDecoder | - | 9 skip | - | - |
| MelWeightMatrix | - | 1 skip | - | - |
| STFT | - | 2 skip | - | - |
| RegexFullMatch | - | 3 skip | - | - |
| StringConcat | - | 5 skip | - | - |
| StringNormalizer | - | 6 skip | - | - |
| StringSplit | - | 6 skip | - | - |
| TfIdfVectorizer | - | 7 skip | - | - |

## com.microsoft Extensions

| Operator | Impl |
|----------|------|
| Attention | Y |
| EmbedLayerNormalization | Y |
| RotaryEmbedding | Y |
| SkipLayerNormalization | Y |
