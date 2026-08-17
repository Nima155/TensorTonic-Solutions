# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Nima155's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/nima155.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Compute AUC (Area Under ROC) | Calculate binary-classification ROC AUC from false-positive and true-positive rates using trapezoidal integration. | https://www.tensortonic.com/problems/auc |
| Chi-Square Test | Run a chi-square independence test on a contingency table using expected counts and the chi-square statistic. | https://www.tensortonic.com/problems/chi2-independence |
| Compute Accuracy, Precision, Recall, F1 | Compute binary accuracy, precision, recall, and F1 score from predicted and true class labels. | https://www.tensortonic.com/problems/classification-metrics |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Expected Calibration Error | Calculate expected calibration error by binning prediction confidence and weighting accuracy-confidence gaps. | https://www.tensortonic.com/problems/expected-calibration-error |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Compute Information Gain for a Split | Compute information gain for a decision-tree split from parent entropy and weighted child entropies. | https://www.tensortonic.com/problems/information-gain |
| Isotonic Regression Calibration | Calibrate prediction scores with isotonic regression while producing a monotonic non-decreasing mapping. | https://www.tensortonic.com/problems/isotonic-calibration |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Implement Majority Class Classifier | Fit a majority-class baseline and predict the most frequent training label for every requested sample. | https://www.tensortonic.com/problems/majority-classifier |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| One-Sample t-Test | Compute a one-sample t-statistic in NumPy using the sample mean, Bessel-corrected deviation, and hypothesized mean. | https://www.tensortonic.com/problems/t-test-one-sample |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Data Augmentation | Implement AlexNet image augmentation operations for deterministic crops, horizontal flips, and intensity changes. | https://www.tensortonic.com/research/alexnet/alexnet-augmentation |
| AlexNet Convolution Layer | Implement an AlexNet convolutional layer with learned filters, bias, stride, padding, and multi-channel outputs. | https://www.tensortonic.com/research/alexnet/alexnet-conv-layers |
| Dropout Regularization | Implement inverted dropout for AlexNet with seeded masks and training-versus-inference behavior. | https://www.tensortonic.com/research/alexnet/alexnet-dropout |
| Overlapping Max Pooling | Implement AlexNet overlapping max pooling with a 3x3 window and stride two across spatial dimensions. | https://www.tensortonic.com/research/alexnet/alexnet-pooling |
| ReLU Activation Function | Implement AlexNet's elementwise ReLU activation, preserving positive values while setting negative values to zero. | https://www.tensortonic.com/research/alexnet/alexnet-relu |
| GAN Discriminator | Implement a GAN discriminator that maps input samples through dense layers to real-versus-fake probabilities. | https://www.tensortonic.com/research/gan/gan-discriminator |
| Complete GAN System | Assemble a complete GAN system that generates samples, scores them with the discriminator, and returns both outputs. | https://www.tensortonic.com/research/gan/gan-full-network |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| GAN Loss Functions | Compute numerically stable binary cross-entropy losses for the GAN generator and discriminator objectives. | https://www.tensortonic.com/research/gan/gan-loss |
| Mode Collapse Detection | Detect GAN mode collapse by measuring diversity across generated samples and flagging low-variance outputs. | https://www.tensortonic.com/research/gan/gan-mode-collapse |
| GAN Training Loop | Implement one GAN training iteration with separate discriminator and generator forward and update steps. | https://www.tensortonic.com/research/gan/gan-training-loop |
| Complete LSTM Cell | Build a complete LSTM cell with forget, input, candidate, cell-state, output, and hidden-state calculations. | https://www.tensortonic.com/research/lstm/lstm-cell |
| Cell State Update | Implement the LSTM cell-state update by combining retained memory with input-gated candidate information. | https://www.tensortonic.com/research/lstm/lstm-cell-state |
| Forget Gate | Implement an LSTM forget gate by combining the previous hidden state and current input with a sigmoid projection. | https://www.tensortonic.com/research/lstm/lstm-forget-gate |
| Complete LSTM Network | Assemble an LSTM sequence forward pass that carries hidden and cell states across every time step. | https://www.tensortonic.com/research/lstm/lstm-full-network |
| Input Gate | Implement the LSTM input gate and candidate activation that control new information written to the cell state. | https://www.tensortonic.com/research/lstm/lstm-input-gate |
| Output Gate | Implement the LSTM output gate and expose the current hidden state from the updated cell memory. | https://www.tensortonic.com/research/lstm/lstm-output-gate |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Backpropagation Through Time | Implement one backpropagation-through-time step using the tanh derivative and hidden-to-hidden weight gradients. | https://www.tensortonic.com/research/rnn/rnn-bptt |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Complete Vanilla RNN | Assemble a vanilla RNN that processes sequences into recurrent hidden states and per-time-step output logits. | https://www.tensortonic.com/research/rnn/rnn-full-network |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| VGG Classifier Head | Build the VGG classifier by flattening spatial features and applying two ReLU hidden layers plus a logits projection. | https://www.tensortonic.com/research/vgg/vgg-classifier |
| VGG Conv Block | Implement a VGG convolutional block as sequential channel projections with ReLU activation at every spatial position. | https://www.tensortonic.com/research/vgg/vgg-conv-block |
| VGG Max Pooling | Implement VGG 2x2 max pooling with stride two while preserving the input batch and channel dimensions. | https://www.tensortonic.com/research/vgg/vgg-maxpool |
| Lasso Regression | Implement Lasso regression with gradient descent, an L1 subgradient penalty on weights, and an unregularized bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-lasso-regression |
| Linear Regression from Scratch | Train linear regression from scratch with mean squared error gradients for weights and bias. | https://www.tensortonic.com/study-plans/cracking-ml/ml-linear-regression-from-scratch |
| Logistic Regression from Scratch | Train binary logistic regression from scratch using sigmoid probabilities, cross-entropy gradients, and gradient descent. | https://www.tensortonic.com/study-plans/cracking-ml/ml-logistic-regression |
| Ridge Regression | Train Ridge regression with gradient descent, L2-regularized weights, and an unregularized bias term. | https://www.tensortonic.com/study-plans/cracking-ml/ml-ridge-regression |
| AdaGrad | Implement the AdaGrad (Adaptive Gradient) optimizer for linear regression with MSE loss using full-batch gradient descent. | https://www.tensortonic.com/study-plans/math-optimization/optim-adagrad |
| RMSProp | Implement the RMSProp optimizer to train a linear regression model using full-batch gradient descent with MSE loss. | https://www.tensortonic.com/study-plans/math-optimization/optim-rmsprop |
| Arange and Linspace | Generate a one-dimensional NumPy sequence using either step-based arange or count-based linspace semantics. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-arange-linspace |
| Basic Indexing | Extract a rectangular NumPy subarray with row and column slice boundaries using standard basic indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-basic-indexing |
| Create Arrays from Lists | Create NumPy arrays from Python lists with the requested dtype and return their values, shape, dimensions, and element count. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-create-array |
| Fancy Indexing | Convert the data to float64 and return the array formed by selecting elements along that axis using integer array indexing. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-fancy-indexing |
| Random Array Generation | Generate seeded float64 NumPy arrays from either a uniform or standard normal distribution. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-random-arrays |
| Reshaping Arrays | Transform a float64 NumPy array with flattening, transposition, or a validated target shape. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-reshape |
| Zeros and Ones | Create a two-dimensional float64 NumPy array of a requested shape filled entirely with zeros or ones. | https://www.tensortonic.com/study-plans/numpy-basics/numpy-zeros-ones |
| Aggregation Functions | Implement Aggregation Functions, and return a dict mapping each function name to a dict of group label to aggregated value. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-aggregation-functions |
| Boolean Indexing | Filter pandas rows by a numeric column threshold and return the matching records with their original column order. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-boolean-indexing |
| Change Data Types | Create a DataFrame, convert the specified column to the target type, and return the dtypes before and after conversion. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-change-dtypes |
| Column Selection | Create a pandas DataFrame from dictionary data and extract one named column as an ordered list. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-column-selection |
| Concatenate DataFrames | Concatenate multiple pandas DataFrames vertically and return the combined records with a reset index. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-concatenate-dataframes |
| Cross Tabulation | Create a DataFrame and compute a cross-tabulation (frequency table) showing how often each combination of values co-occurs. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-cross-tabulation |
| Data Types Overview | Create a pandas DataFrame and report each column dtype together with counts for every unique dtype. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-data-types |
| Drop Duplicates | Create a DataFrame, remove duplicate rows, and return the cleaned result along with counts of rows before and after deduplication. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-drop-duplicates |
| GroupBy Basics | Create a DataFrame and compute the sum, mean, and count of the value column for each group. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-groupby-basics |
| Handle Missing Values | Create a pandas DataFrame, count missing entries per column, and replace every null with a supplied fill value. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-handle-missing |
| Head and Tail Operations | Create a pandas DataFrame and return the requested first and last rows as record-oriented dictionaries. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-head-tail |
| Inspect DataFrame Shape | Create a DataFrame and return its structural properties: row count, column count, column names, data types, and total number of values. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-inspect-shape |
| Loc vs iLoc | Create a DataFrame and use positional indexing to extract: the single element, the full row, and the full column. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-loc-iloc |
| Melt Wide to Long | Reshape a pandas DataFrame from wide to long format using selected identifier and value columns. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-melt-wide-to-long |
| Merge DataFrames | Use two dictionaries of column data and a key column present in both, create two DataFrames and merge them on the key column using a specified join type. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-merge-dataframes |
| Multi-Column Selection | Create a pandas DataFrame and select an ordered subset of named columns without changing row order. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-multi-column-selection |
| Multi-Level GroupBy | Create a DataFrame, group by all specified columns, apply the aggregation, and return the result as a flat table. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-multi-level-groupby |
| Pivot Tables | Build a pandas pivot table with selected index, columns, values, aggregation, and zero-filled missing combinations. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-pivot-tables |
| Create DataFrame from Dict | Create a pandas DataFrame from dictionary data and report its records, shape, and ordered column names. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-read-csv |
| Rename Columns | Rename selected pandas DataFrame columns from an old-to-new mapping and return the updated records. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-rename-columns |
| Replace Values | Create a DataFrame, replace all occurrences of the old value with the new value in the specified column, and count how many replacements were made. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-replace-values |
| Resetting Index | Set a pandas column as the index, then restore the default integer index while retaining the original values. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-resetting-index |
| Setting Index | Set a named pandas DataFrame column as the index and report the resulting records and index metadata. | https://www.tensortonic.com/study-plans/pandas-basics/pandas-setting-index |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/nima155)
<!-- tensortonic:end -->
