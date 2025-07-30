from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mysql.connector

# Connect to the running PAGAN MySQL (inside Docker)
db_config = {
    "host": "127.0.0.1",
    "user": "pagan_sys",
    "password": "pagan_password",
    "database": "pagan",
    "port": 3306
}

class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/upload":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                annotations = json.loads(post_data.decode('utf-8'))

                cnx = mysql.connector.connect(**db_config)
                cursor = cnx.cursor()

                for annotation in annotations:
                    time_value = annotation.get("time")
                    value_value = float(annotation.get("value", 0))
                    cursor.execute(
                        "INSERT INTO annotations (time, value) VALUES (%s, %s)",
                        (time_value, value_value)
                    )

                cnx.commit()
                cursor.close()
                cnx.close()

                print(f"Inserted {len(annotations)} annotations into MySQL.")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Annotations uploaded and stored in MySQL.")
            except Exception as e:
                print(f"MySQL error: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"MySQL insertion failed.")
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            with open('.' + self.path, 'rb') as file:
                content = file.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SimpleHandler)
    print("Server running at http://localhost:8000")
    httpd.serve_forever()
