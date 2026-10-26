"""Loopback static server with single byte ranges for seeking local MP4 clips."""
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
import re


def byte_range(header, size):
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if match is None or size <= 0:
        raise ValueError('Invalid byte range')
    first, last = match.groups()
    if not first:
        if not last or int(last) <= 0:
            raise ValueError('Invalid suffix range')
        start, end = max(0, size-int(last)), size-1
    else:
        start, end = int(first), min(int(last), size-1) if last else size-1
    if start >= size or end < start:
        raise ValueError('Unsatisfiable byte range')
    return start, end


class ReviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self._byte_range = None
        path = Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(Path(self.directory).resolve()):
            self.send_error(403)
            return None
        if not path.is_file() or path.suffix.lower() != '.mp4':
            return super().send_head()
        size = path.stat().st_size
        header = self.headers.get('Range')
        if header:
            try:
                self._byte_range = byte_range(header, size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
        file = path.open('rb')
        self.send_response(206 if header else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        if self._byte_range:
            start, end = self._byte_range
            file.seek(start)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.send_header('Content-Length', str(end-start+1))
        else:
            self.send_header('Content-Length', str(size))
        self.end_headers()
        return file

    def copyfile(self, source, outputfile):
        try:
            if self._byte_range is None:
                return super().copyfile(source, outputfile)
            start, end = self._byte_range
            remaining = end-start+1
            while remaining:
                block = source.read(min(64*1024, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browser cancelled an earlier buffer request while seeking.
