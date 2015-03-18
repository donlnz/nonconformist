# nonconformist

Python implementation of the conformal prediction framework [1].

Primarily to be used as an extension to the scikit-learn library.

# Dependencies

core: numpy; tested under Python 2.7

examples: sklearn

# Known issues

* Conformal classifiers only support numerical classes (0, 1, ...). This *might* change in the future.

[1] Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer Science & Business Media, 2005.