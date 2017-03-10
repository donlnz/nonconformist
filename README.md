# nonconformist

Python implementation of the conformal prediction framework [1].

Primarily to be used as an extension to the scikit-learn library.

API documentation: http://donlnz.github.io/nonconformist/

# Dependencies
Tested under Python 2.7.

Requires numpy, scipy, pandas and scikit-learn.

# Known issues

* Conformal classifiers only support numerical classes (0, 1, ...). This *might* change in the future.

# TODO

* Exchangeability testing [2].
* Transductive conformal prediction [1].
* Interpolated p-values [3,4].
* Conformal prediction trees [5].

[1] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. Springer Science & Business Media.

[2] Fedorova, V., Gammerman, A., Nouretdinov, I., & Vovk, V. (2012). Plug-in martingales for testing exchangeability on-line. In Proceedings of the 29th International Conference on Machine Learning (ICML-12) (pp. 1639-1646).

[3] Carlsson, L., Ahlberg, E., Boström, H., Johansson, U., Linusson, & H. (2015). Modifications to p-values of Conformal Predictors. In Proceedings of the 3rd International Symposium on Statistical Learning and Data Sciences (SLDS 2015). (In press).

[4] Johansson, U., Ahlberg, E., Boström, H., Carlsson, L., Linusson, H., Sönströd, C. (2015). Handling Small Calibration Sets in Mondrian Inductive Conformal Regressors. In Proceedings of the 3rd International Symposium on Statistical Learning and Data Sciences (SLDS 2015). (In press).

[5] Johansson, U., Sönströd, C., Linusson, H., & Boström, H. (2014, October). Regression trees for streaming data with local performance guarantees. In Big Data (Big Data), 2014 IEEE International Conference on (pp. 461-470). IEEE.