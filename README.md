# Organization of the somatosensory cortex in ASD

Code for the manuscript ''*Fine-grained topographic organization within somatosensory cortex during resting-state and emotional face-matching task and its association with ASD traits.*''

We used code from Haak et al. (2018) that can be found  [here](https://github.com/koenhaak/congrads) to get connectopies on MNI space, the TSM coefs that reconstruct them and their projection maps.

**MEMO_run.py** is the main file where the code for exporting figure 2 (main text), heatmaps and model selection figure (supplementary material) can be found. 

The surface maps of average connectopies and projection maps have been made with 
[Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench#:~:text=Connectome%20Workbench%20is%20an%20open,by%20the%20Human%20Connectome%20Project.).

*Code has run with Python 3.6 and the following packages:  
+matplotlib\==3.0.3  
+nibabel\==2.3.4  
+nilearn\==0.5.0  
+numpy\==1.16.2  
+pandas\==0.24.2  
+scikit-learn\==0.20.3  
+scipy\==1.2.1  
+seaborn\==0.11.0  
+statsmodels\==0.10.1*

