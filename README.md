<h1>machine_learning</h1>
<p>In this repository i am using python to practice machine learning algorithms&nbsp;<br />
  This exercise is done part of Machine Learning Course at the Haifa University (2016/2017)</p>

<blockquote>

<p><strong>1. ex1.py</strong>&nbsp;:&nbsp;</p>
  <p>    See video (output of this program):&nbsp;<a href="https://youtu.be/0QvxtB3d8h4">https://youtu.be/0QvxtB3d8h4</a>&nbsp;<br />
  </p>
  <p><strong>Algorithm</strong>: Neural Network&nbsp;<br />
    <strong>Dataset</strong>: Iris DS<br />
    <strong>Descriptipon</strong>: The exercise solves two problems using Neural Network: the OR boolean statement and linear separation of the Iris  dataset.  &nbsp;<br />
    <strong>Parameters</strong>: </p>
  <ul>
    <li>plots output folder (Default is c:/temp)</li>
    <li>ratio</li>
    <li>Two features selected: &quot;sepal length&quot;, &quot;sepal width&quot;</li>
    <li> Two classes of Iris where selected: &quot;Iris-setosa&quot;, &quot;Iris-virginica&quot;</li>
  </ul>
  <p><br />
    <strong>Algorithm</strong>: Neural Network&nbsp;<br />
    <strong>Dataset</strong>: Iris DS<br />
  </p>
  
  <p>anyway, these can be changed easily in the python program to check different results of different parameter variations.
    I randomly split the data to training and testing (based on the vaue in the first column) and i got three different results, but they all are close together:</p>
  <p>&nbsp;</p>
  
  <table bordercolor="#000000">
    <tr>
    <th width="66" bordercolor="#000000">Ratio</th>
    <th width="68" bordercolor="#000000">Iterations</th>
    <th width="103" bordercolor="#000000">theta</th>
    <th width="96" bordercolor="#000000">Accuracy</th>
    </tr>
    <tr>
    <td bordercolor="#000000"><div align="center">0.6</div></td>
    <td bordercolor="#000000"><div align="center">16</div></td>
    <td bordercolor="#000000">-4.8 , 7.2</td>
    <td bordercolor="#000000">0.77</td>
    </tr>
    <tr>
    <td bordercolor="#000000"><div align="center">0.6</div></td>
    <td bordercolor="#000000"><div align="center">19</div></td>
    <td bordercolor="#000000">-5.6 , 9.1</td>
    <td bordercolor="#000000">0.9</td>
    </tr>
    <tr>
    <td bordercolor="#000000"><div align="center">0.6</div></td>
    <td bordercolor="#000000"><div align="center">259</div></td>
    <td bordercolor="#000000">-51.5 , 76.9</td>
    <td bordercolor="#000000">0.8</td>
    </tr>       
    <tr>
    <td bordercolor="#000000"><div align="center">0.6</div></td>
    <td bordercolor="#000000"><div align="center">164</div></td>
    <td bordercolor="#000000">-40.1 , 61.5</td>
    <td bordercolor="#000000">0.8919</td>
    </tr>   
    <tr>
    <td bordercolor="#000000"><div align="center">0.8</div></td>
    <td bordercolor="#000000"><div align="center">139</div></td>
    <td bordercolor="#000000">-29.29 , 44.30</td>
    <td bordercolor="#000000">0.8</td>
    </tr>      
    <tr>
    <td bordercolor="#000000"><div align="center">0.8</div></td>
    <td bordercolor="#000000"><div align="center">280</div></td>
    <td bordercolor="#000000">-51.8 , 76.0</td>
    <td bordercolor="#000000">0.875</td>
    </tr>              
</table>
  

  <p>The number of iterations till it converges is not deteministic since we randomly split between training dataset and test dataset.</p>
  
  </blockquote>
<blockquote>
  <p><strong>2. ex2.py&nbsp;: Comparison between Machine Learning algorithms&nbsp;</strong><br />
    This exercise is targeting to compare between the two algorithms of SVM and Neural Network on different datasets.&nbsp;<br />
    The program can be split into the following major parts:&nbsp;<br />
  </p>
  <ol>
    <li>SVM learning process:</li>
    <ol>
      <li>Run SVM model on the dataset using different combinations of gamma and C, and find the best combination</li>
      <li>For doing this I am running loops on gamma and C with big steps. In the beginging I chose a big range (1 to 1000) with big steps (of 100), then later on I zoom into smaller range with smaller steps.&nbsp;<br />
        I stop either if I reach 10 times or if there are no meaningful improvement on the score anymore</li>
    </ol>
    <li>Neural Network Learning process. We need to find the best combination of .<br />
        <br />
      Note: In this exercise we set the value of learning rate and we keep it constant</li>
    <li>Comparison between the &lsquo;best&rsquo; SVM model and the &lsquo;best&rsquo; NN model.</li>
  </ol>

</blockquote>
