# Space-split sensitivity analysis on chaotic systems

This program is a Monte Carlo implementation of the [linear response formula][1] for chaotic systems due to David Ruelle.
The output is the derivative of the expectation with respect to the SRB measure, of a user-defined scalar 
function `objective`, with respect to scalar parameter `s`.

For the derivation of S3, please refer to [this preprint][2].

[1]: https://link.springer.com/article/10.1007/s002200050134
[2]: https://arxiv.org/pdf/2002.04117.pdf



