# machine_learning
practicing some machine learning algorithms
The exercise are done part of Machine Learning Course at the Haifa University (2016/2017)

1. <b><u>ex1.py</u></b> : 
    <br>See video (output of this program): https://youtu.be/BcnmQQMt6bs
    <br><b>Algorithm</b>: Neural Network
    <br><b>Dataset</b>: Iris DS
    
    <br><b>Description</b>: The exercise solves two problems using Neural Network: the OR boolean statement and linear separation of the Iris  dataset. 
    
    <br><b>Parameters</b>:
    - plots output folder (Default is c:/temp)
    - ratio
    - Two features selected: "sepal length", "sepal width"
    - Two classes of Iris where selected: "Iris-setosa", "Iris-virginica"
    
    <br>anyway, these can be changed easily in the python program to check different results of different parameter variations.
    <br>I randomly split the data to training (60%) and testing (40%) and i got three different results: 
    <table>
        <tr>
        <td>Iterations</td>
        <td>theta</td>
        <td>Accuracy</td>
        </tr>
        <tr>
        <td>16</td>
        <td>-4.8 , 7.2</td>
        <td>0.77</td>
        </tr>
        <tr>
        <td>19</td>
        <td>-5.6 , 9.1</td>
        <td>0.9</td>
        </tr>
        <tr>
        <td>259</td>
        <td>-51.5 , 76.9</td>
        <td>0.8</td>
        </tr>       
        <tr>
        <td>164</td>
        <td>-40.1 , 61.5</td>
        <td>0.8919</td>
        </tr>       
    </table>
    
    
    <br>
    The number of iterations till it converges is not deteministic since we randomly split between training dataset and test dataset.
    
    
