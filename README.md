# Space-split sensitivity analysis on chaotic systems

This program is a Monte Carlo implementation of the [linear response formula][1] for chaotic systems due to David Ruelle.
The output is the derivative of the expectation with respect to the SRB measure, of a user-defined scalar 
function `objective`, with respect to scalar parameter `s`.

For the derivation of S3, please refer to [this preprint][2].

[1]: https://link.springer.com/article/10.1007/s002200050134
[2]: https://arxiv.org/pdf/2002.04117.pdf

The differential CLV method computes the directional derivatives of covariant Lyapunov vectors in their own directions. This is a byproduct of the S3 algorithm, see <a href="https://zenodo.org/badge/latestdoi/240020405"><img src="https://zenodo.org/badge/240020405.svg" alt="DOI"></a>


