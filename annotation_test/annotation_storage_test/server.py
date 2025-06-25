from http.server import BaseHTTPRequestHandler, HTTPServer

class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        with open("annotation.json", "wb") as f:
            f.write(post_data)

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Annotation received and saved.")

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            with open('.' + self.path, 'rb') as file:
                content = file.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(content)
        except:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SimpleHandler)
    print("Server running at http://localhost:8000")
    httpd.serve_forever()
