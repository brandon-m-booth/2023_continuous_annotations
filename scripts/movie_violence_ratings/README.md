# Getting Ratings from CommonSense Media
Movie ratings from CSM are easily obtained from their website: [https://www.commonsensemedia.org/](https://www.commonsensemedia.org/). Since we aimed to find movies with shorter running times for our study, we used the CSM API to access all rated movies and then select the ones matching our research goals. For completeness, we provide this code and the following steps we followed for our case study:

1. Get an [API key from CSM following their instructions](https://www.commonsensemedia.org/developers/api-overview)
1. From mica-commonsense-api, run `python run.py API_KEY` to get a pickle file with all CSM ratings
1. Run `python gen_movie_violence_csv.py --csm_pkl mica-commonsense-api/commonsense.pkl --output_csv csm_violence.csv` to turn the pickle file contents into a csv file
1. Select the desired movies from the csv file based on research criteria (e.g., for our study: one movie from 2018-2019 per violence rating with the shortest running time)
