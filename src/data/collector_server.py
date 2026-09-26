from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import base64
import binascii
import json
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

HOST = "127.0.0.1"
PORT = 8000


class CollectorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        request = urlparse(self.path)

        if request.path == "/sample-counts":
            self.handle_sample_counts(request)
            return

        super().do_GET()

    def do_POST(self):
        if self.path != "/save-sample":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers["Content-Length"])
            request_body = self.rfile.read(content_length)

            sample = json.loads(request_body)

            writer_id = sample["writer_id"]
            digit = sample["digit"]
            image_data = sample["image"]

            self.validate_writer_id(writer_id)

            if digit not in "0123456789" or len(digit) != 1:
                raise ValueError("Digit must be from 0 to 9.")

            prefix = "data:image/png;base64,"

            if not image_data.startswith(prefix):
                raise ValueError("Image must be a PNG.")

            encoded_image = image_data[len(prefix):]
            image_bytes = base64.b64decode(encoded_image, validate=True)

            digit_dir = RAW_DATA_DIR / writer_id / digit
            digit_dir.mkdir(parents=True, exist_ok=True)

            existing_numbers = [
                int(path.stem)
                for path in digit_dir.glob("*.png")
                if path.stem.isdigit()
            ]

            sample_number = max(existing_numbers, default=0) + 1

            filename = f"{sample_number:04d}.png"
            output_path = digit_dir / filename

            output_path.write_bytes(image_bytes)

            self.send_json(
                200,
                {
                    "saved": True,
                    "file": str(output_path.relative_to(PROJECT_ROOT)),
                    "sample_number": sample_number,
                    "counts": self.get_sample_counts(writer_id),
                },
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            binascii.Error,
        ) as error:
            self.send_json(
                400,
                {
                    "saved": False,
                    "error": str(error),
                },
            )

    def handle_sample_counts(self, request):
        try:
            query = parse_qs(request.query)
            writer_id = query.get("writer_id", [""])[0]

            self.validate_writer_id(writer_id)

            self.send_json(
                200,
                {
                    "writer_id": writer_id,
                    "counts": self.get_sample_counts(writer_id),
                },
            )

        except ValueError as error:
            self.send_json(
                400,
                {
                    "error": str(error),
                },
            )

    def validate_writer_id(self, writer_id):
        if not writer_id:
            raise ValueError("Writer ID is required.")

        if not writer_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Writer ID contains invalid characters.")

    def get_sample_counts(self, writer_id):
        counts = {}

        for digit in "0123456789":
            digit_dir = RAW_DATA_DIR / writer_id / digit

            if digit_dir.exists():
                counts[digit] = len(list(digit_dir.glob("*.png")))
            else:
                counts[digit] = 0

        return counts

    def send_json(self, status_code, data):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()

        self.wfile.write(response)


def main():
    os.chdir(PROJECT_ROOT)

    server = ThreadingHTTPServer((HOST, PORT), CollectorHandler)

    print(f"Digit collector running at http://localhost:{PORT}/interface/collect.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
