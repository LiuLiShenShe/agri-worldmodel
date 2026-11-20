"""Pinned 4TU v1 inputs and bounded, cached ZIP64 member reads."""

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/source-cache'
DOIS = {24: 'fa102772-32db-4b30-bace-12f2016722ce',
        23: 'e1ee9de9-6ce9-4502-a37c-34b5b1372bed'}


@dataclass(frozen=True)
class Source:
    year: int
    file_id: str
    name: str
    size: int
    md5: str

    @property
    def url(self):
        return f'https://data.4tu.nl/file/{DOIS[self.year]}/{self.file_id}'


SOURCES = {
    'readme24': Source(24, '4cc2f511-9eb4-4cd3-99b9-97a6e106f3e1', 'README.md', 3715, 'f22125b584f3e25ae80ed87e2ba72de1'),
    'series24': Source(24, '14a4a8cd-e081-4dbd-94e3-cd7b332ce120', 'autonomous_greenhouse_challenge4_timeseries.zip', 6014133, 'c4da1b2696c4a1e5ea5744f2482b0f93'),
    'camera24': Source(24, '818e2966-7015-432d-aa7c-1b9158c011cf', 'autonomous_greenhouse_challenge4_canopy_camera.zip', 31416872168, '671a1c6e2b2e1627f821a36e431ceb28'),
    'series23': Source(23, '728b4651-2127-4fd1-a6dc-9abf854e0552', '4th_autonomous_greenhouse_challenge_dwarf_tomato_pretrial_Timeseries.zip', 4224101, '8918ef8423bd0e907739b89a935f0eb0'),
    'canopy23': Source(23, '547a07d7-ffe8-438e-8bb2-8d3cdea9daff', '4th_autonomous_greenhouse_challenge_dwarf_tomato_pretrial_canopy_cam.zip', 4499363312, '47dc897f763ebc14d8be5d5f9c8c5981'),
    'single23': Source(23, '923e3ab0-011f-432c-9039-2151e2c4a6f9', '4th_autonomous_greenhouse_challenge_dwarf_tomato_pretrial_single_plant_cam.zip', 7456535483, '81d46dddbddb6519736fedfa8d73f694'),
}
SMALL = {'readme24': 'agc2024_README.md', 'series24': 'agc2024_timeseries.zip', 'series23': 'agc2023_timeseries.zip'}


def session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=2, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)))
    return s


def small(key):
    source = SOURCES[key]
    path = CACHE / SMALL[key]
    if path.exists():
        data = path.read_bytes()
    else:
        with session() as net:
            r = net.get(source.url, timeout=(15, 45))
            r.raise_for_status()
            data = r.content
    if len(data) != source.size or hashlib.md5(data).hexdigest() != source.md5:
        raise ValueError(f'{key}: size/MD5 mismatch, check {path}')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


class RangeReader(io.RawIOBase):
    """Expose a pinned remote object to stdlib zipfile; never accept a full response."""

    def __init__(self, key):
        self.source = SOURCES[key]
        self.net = session()
        self.pos = 0
        self.tail_start = max(0, self.source.size - 1024 * 1024)
        tail_path = CACHE / 'range' / f'{key}.tail'
        tail_path.parent.mkdir(parents=True, exist_ok=True)
        self.tail = tail_path.read_bytes() if tail_path.exists() else None
        if self.tail is None or len(self.tail) != self.source.size - self.tail_start:
            self.tail = self.fetch(self.tail_start, self.source.size - 1)
            tail_path.write_bytes(self.tail)

    def fetch(self, first, last):
        expected = f'bytes {first}-{last}/{self.source.size}'
        with self.net.get(self.source.url, headers={'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity'},
                          stream=True, timeout=(15, 35)) as r:
            if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                raise OSError(f'{self.source.name}: expected {expected}; got HTTP {r.status_code} {r.headers.get("Content-Range")}')
            data = r.content
        if len(data) != last - first + 1:
            raise OSError(f'{self.source.name}: truncated range {expected}')
        return data

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = max(0, (0, self.pos, self.source.size)[whence] + offset)
        return self.pos

    def read(self, n=-1):
        end = self.source.size if n < 0 else min(self.source.size, self.pos + n)
        if end <= self.pos:
            return b''
        if end - self.pos > 32 * 1024 * 1024:
            raise OSError('Oversize Range read refused')
        start = self.pos
        data = self.tail[start - self.tail_start:end - self.tail_start] if start >= self.tail_start else self.fetch(start, end - 1)
        self.pos = end
        return data

    def close(self):
        self.net.close()
        super().close()


class Archive:
    def __init__(self, key):
        self.key = key
        self.reader = RangeReader(key)
        self.z = zipfile.ZipFile(self.reader)
        self.names = self.z.namelist()

    def close(self):
        self.z.close()
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, name):
        info = self.z.getinfo(name)
        safe_name = Path(name)
        if safe_name.is_absolute() or '..' in safe_name.parts:
            raise ValueError(f'Unsafe member: {name}')
        path = CACHE / 'members' / self.key / safe_name
        if path.is_file():
            data = path.read_bytes()
        else:
            # One bounded request for the local header and compressed payload avoids
            # repeated HTTP round-trips by ZipExtFile on a remote object.
            end = min(self.reader.source.size, info.header_offset + info.compress_size + 16384)
            data_range = self.reader.fetch(info.header_offset, end - 1)
            old = self.z.fp
            class MemberBuffer(io.BytesIO):
                def seek(self, offset, whence=0):
                    return super().seek(offset - info.header_offset if whence == 0 else offset, whence)

                def tell(self):
                    return super().tell() + info.header_offset

            self.z.fp = MemberBuffer(data_range)
            try:
                data = self.z.read(info)
            finally:
                self.z.fp = old
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if len(data) != info.file_size or zipfile.crc32(data) & 0xffffffff != info.CRC:
            raise ValueError(f'{name}: size/CRC mismatch')
        return data

    def evidence(self, name, data):
        return dict(archive=self.key, member=name, bytes=len(data),
                    crc32=f'{self.z.getinfo(name).CRC:08x}', sha256=hashlib.sha256(data).hexdigest())
