# machine_learning
In this repository i am using python to practice machine learning algorithms
<br>
This exercise is done part of Machine Learning Course at the Haifa University (2016/2017)

<b>1. <u>ex1.py</u></b> : 
    <br>See video (output of this program): https://youtu.be/0QvxtB3d8h4 
    <br><b>Algorithm</b>: Neural Network
    <br><b>Dataset</b>: Iris DS
    
    <br><b>Description</b>: The exercise solves two problems using Neural Network: the OR boolean statement and linear separation of the Iris  dataset. 
    
    <br><b>Parameters</b>:
    - plots output folder (Default is c:/temp)
    - ratio
    - Two features selected: "sepal length", "sepal width"
    - Two classes of Iris where selected: "Iris-setosa", "Iris-virginica"
    
    <br>anyway, these can be changed easily in the python program to check different results of different parameter variations.
    <br>I randomly split the data to training and testing (based on the vaue in the first column) and i got three different results, but they all are close together: 
    <table>
        <tr>
        <td>Ratio</td>
        <td>Iterations</td>
        <td>theta</td>
        <td>Accuracy</td>
        </tr>
        <tr>
        <td>0.6</td>
        <td>16</td>
        <td>-4.8 , 7.2</td>
        <td>0.77</td>
        </tr>
        <tr>
        <td>0.6</td>
        <td>19</td>
        <td>-5.6 , 9.1</td>
        <td>0.9</td>
        </tr>
        <tr>
        <td>0.6</td>
        <td>259</td>
        <td>-51.5 , 76.9</td>
        <td>0.8</td>
        </tr>       
        <tr>
        <td>0.6</td>
        <td>164</td>
        <td>-40.1 , 61.5</td>
        <td>0.8919</td>
        </tr>   
        <tr>
        <td>0.8</td>
        <td>139</td>
        <td>-29.29 , 44.30</td>
        <td>0.8</td>
        </tr>      
        <tr>
        <td>0.8</td>
        <td>280</td>
        <td>-51.8 , 76.0</td>
        <td>0.875</td>
        </tr>              
    </table>
    
    
    <br>
    The number of iterations till it converges is not deteministic since we randomly split between training dataset and test dataset.
    
<b>2. <u>ex2.py</u></b> : Comparison between Machine Learning algorithms
    <br>
    This exercise is targeting to compare between the two algorithms of SVM and Neural Network on different datasets. 
    <br>
    The program can be split into the following major parts:
              <br><ol>
          <li>SVM learning process:    </li>
          <ol style="list-style-type: lower-alpha; padding-bottom: 0;">
            <li style="margin-left:2em">
               Run SVM model on the dataset using different combinations of gamma and C, and find the best combination   </li>
            <li style="margin-left:2em">
               For doing this I am running loops on gamma and C with big steps. In the beginging I chose a big range (1 to 1000) with big steps (of 100), then later on I zoom into smaller range with smaller steps.    <br>
                    I stop  either if I reach 10 times or if there are no meaningful improvement on the score anymore    </li>
          </ol>
          <li>Neural Network Learning process. We need to find the best combination of <layer, learning rate>.     
   <br>
          Note: In this exercise we set the value of learning rate and we keep it constant    </li>
          <li>
          Comparison between the ‘best’ SVM model and the ‘best’ NN model.     </li>
          </ol>
          


