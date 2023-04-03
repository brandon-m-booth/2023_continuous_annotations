import requests
import json

class CommonSenseMediaAPI:
	
	def __init__(self,
				 api_key, 
				 base_url = "https://api.commonsense.org",
				 version = "v2"):

		self.api_key = api_key
		self.headers = {"x-api-key": api_key}
		self._set_urls(base_url, version)

	def _set_urls(self, base_url, version):
		"""inits base urls"""
		self.urls = {}

		# System ping
		self.urls['ping'] = "system/ping"

		# Review/browse
		self.urls['reviews'] = "review/browse"

		for key, url in self.urls.items():
			self.urls[key] = "{}/{}/{}".format(base_url, version, url)
		
	def ping(self):
		"""Used for testing a connection to and availability of the API."""
		r = requests.get(self.urls['ping'],
						 headers = self.headers)
		if r.ok:
			return True

		return False

	def get_movie_reviews(self, limit = None):
		"""Gets all reviews based on submitted filters. 
		Results are returned in order of date last updated, 
		from most recent to oldest."""
		payload = {'channel':'movie'}
		if limit:
			payload['limit'] = limit
		r = requests.get(self.urls['reviews'],
						 params = payload,
						 headers = self.headers)
		if r.ok:
			return r.json()
		else:
			r.raise_for_status()

