# autokit - machine learning for busy people

*this description will be updated*

The method is based on the [hyperopt](https://github.com/hyperopt/hyperopt) and [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) packages to pose the automatic machine learning problem as a hyperparameter optimization problem.

The approach  extends the hyperopt-sklearn model to include additional learning model selection that is able to determine admissible learning models given the problem description. 

It also includes different pre-processing approaches in the hyperparameter search space. This allows us to not only tune individual approaches, but also enrich the data with different representation, such as clustering to have a lower dimensional representation, or kernel approximation to cover non-linearities in the model. 
