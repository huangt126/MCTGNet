# MCTGNet
A Multi-Scale Convolution and Hybrid AttentionNetwork for Robust Motor Imagery EEG Decoding
\abstract{Motor imagery (MI) EEG decoding is a key application in brain-computer interface (BCI) research. In cross-session scenarios, the generalization and robustness of decoding models are particularly challenged due to the complex nonlinear dynamics of MI-EEG signals in both temporal and frequency domains, as well as distributional shifts across different recording sessions.
	While multi-scale feature extraction is a promising approach for generalized and robust MI decoding,
	conventional classifiers(\eg, multilayer perceptrons)struggle to perform accurate classification when 
	confronted with high-order, nonstationary feature distributions, 
	which has become a major bottleneck for improving decoding performance.
	To address this issue, we propose an end-to-end decoding framework, MCTGNet, 
	whose core idea is to formulate the classification process as a high-order function approximation task that jointly models both task labels and feature structures.
	By introducing a group rational Kolmogorov–Arnold Network (GR-KAN), the system enhances generalization and robustness under cross-session conditions.
	Experiments on the BCI Competition IV 2a and 2b datasets demonstrate that MCTGNet achieves average classification accuracies of \textbf{88.93\%} and \textbf{91.42\%}, respectively, outperforming \textcolor{red}{state-of-the-art} methods by \textbf{3.32\%} and \textbf{1.83\%}. \textcolor{red}{The source code has been released at \url{https://github.com/huangt126/MCTGNet}.}
}

We would like to thank all those who have contributed to this work. Thank you!
