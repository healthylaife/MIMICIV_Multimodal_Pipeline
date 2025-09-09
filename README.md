# MIMICIV_Multimodal_Pipeline

### Repository Structure

- **mainMultimodalPipeline.ipynb**
	is the main file to interact with the pipeline. It provides step-step by options to extract and pre-process cohorts.
- **./data**
	consists of all data files stored during pre-processing
	- **./cohort**
		consists of files saved during cohort extraction
	- **./features**
		consist of files containing features data for all selected features.
	- **./summary**
		consists of summary files for all features.
	 	It also consists of file with list of variables in all features and can be used for feature selection.
	- **./dict**
		consists of dictionary structured files for all features obtained after time-series representation
	- **./output**
		consists output files saved after training and testing of model. These files are used during evaluation.
- **./mimic-iv-1.0**
- **./mimic-iv-2.0**
- **./mimic-iv-3.1**
	consist of files downloaded from MIMIC-IV website.
- **./saved_models**
	consists of models saved during training.
- **./preprocessing**
	- **./day_intervals_preproc**
		- **day_intervals_cohort.py** file is used to extract samples, labels and demographic data for cohorts.
		- **disease_cohort.py** is used to filter samples based on diagnoses codes at time of admission
	- **./hosp_module_preproc**
		- **feature_selection_hosp.py** is used to extract, clean and summarize selected features for non-ICU data.
		- **feature_selection_icu.py** is used to extract, clean and summarize selected features for ICU data.
- **./model**
	- **train.py**
		consists of code to create batches of data according to batch_size and create, train and test different models.
	- **Mimic_model.py**
		consist of different model architectures.
	

### How to use the pipeline?
- After downloading the repo, open **mainMultimodalPipeline.ipynb**.
- **mainMultimodalPipeline.ipynb**, contains sequential code blocks to extract, preprocess, model and train MIMIC-IV EHR data.
- Follow each code bloack and read intructions given just before each code block to run code block.
- Follow the exact file paths and filenames given in instructions for each code block to run the pipeline.
- For evaluation module, clear instructions are provided on how to use it as a standalone module.
