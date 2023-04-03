from commonsense import CommonSenseMediaAPI
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('API_KEY', help = "API key for CSM")
args = parser.parse_args()


# Construct CSM api
api = CommonSenseMediaAPI(api_key = args.API_KEY)

print("Getting movie reviews from API")
j = api.get_movie_reviews()

print("Saving to pickle")
pickle.dump(j, open("commonsense.pkl", "wb"))

print("Done")
