import pandas as pd 



def description(dataset, ds_name):
    """
    Argument: (dataframe, operational name of the dataframe)
    They need to have fields:
    ClientID
    Site
    EpisodeOutcome
    StudyInstanceUID
    Manufacturer
    Model
    PatientAgeYears
    LesionSide

    Returns: 
    descriptive stats for the dataset
    """
    dataset.name = ds_name


    number_patients = dataset['ClientID'].nunique()
    number_studies = dataset['StudyInstanceUID'].nunique()
    number_entries = dataset['SOPInstanceUID'].nunique()


    print(f'In the dataset {dataset.name} there are: \n {number_patients} unique patients,   \n {number_studies} unique studies and \n {number_entries} images')

    print('-------------------------- \n')

    dataset.loc[(dataset.PatientAgeYears < 55),  'AgeGroup'] = '<55'
    dataset.loc[(dataset.PatientAgeYears >= 55)& (dataset.PatientAgeYears < 65),  'AgeGroup'] = '55-65'
    dataset.loc[(dataset.PatientAgeYears >= 65),  'AgeGroup'] = '65+'
    age_listing = dataset.groupby(['StudyInstanceUID'], as_index=False)['AgeGroup'].agg('first')
    age_distrib =age_listing.groupby('AgeGroup').count()
    age_distrib 


    print(f'In the dataset {dataset.name} we have the following age counts (one count per StudyInstanceUID):')
    print(age_distrib, '\n')
    print(age_distrib/age_distrib.sum())

    print('-------------------------- \n')


    for col in dataset[['Site', 'Model', 'Manufacturer']]:
        study_grouping = dataset.groupby(['StudyInstanceUID'], as_index=False)[col].agg('first')

        distribution =study_grouping[col].value_counts(dropna=False)
        distribution_percent = (study_grouping[col].value_counts(dropna=False))/(study_grouping[col].value_counts(dropna=False).sum())

        print(f'The patients in {dataset.name} are distributed among the following {col} (one count per StudyInstanceUID):')
        print(distribution, '\n')
        print(distribution_percent)
        print('--------------------------\n')
    