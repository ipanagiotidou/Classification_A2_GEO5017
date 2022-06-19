The code consists of three modules:
- The 'classification.py' which contains the core code that implements the classification algorithms. 
- The 'calculate_features' which load the point cloud of the different objects and calculates the different features. 
- The 'plot_learning_curve' which helps desinging the box plot of the effect of Ratio to the Overall Accuracy. 

The algorithm should run when running the 'classification.py' which invokes the two other modules.

P.S. The part of the code that ensures that the ratio (train-test) is enforced to each one of the object categories separately 
is deactivated. If not deactivated the program informs the user that should rerun the code in case the condition is not satisfied. 
To make your life easier when running the code, this part is deactivated. However, my results were produced with the part activated.
In the Conclusion section of my Report, I discuss this inefficacy in more details. 
You can find this part in the 'classification.py' module under the title in blue: '# TODO: DEACTIVATE WHEN FINISHED'