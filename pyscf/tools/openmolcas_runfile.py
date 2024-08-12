import numpy as np

# see OpenMolcas source mkrun.F90 and runfile_data.F90 for RunFile data structure

def split_str(s, n=16):
    """Split a string into fixed-length substrings"""
    return [s[i: i+n].strip() for i in range(0, len(s), n)]

class runfile_reader:
    def __init__(self, filename):
        f = open(filename, 'rb')
        toc = np.fromfile(f, int, 9)
        n_items = toc[3]
        toc = toc[4:]
        toc *= 512
        label_offset, offset_offset, len_offset, \
                maxlen_offset, type_offset = toc

        labels = []
        f.seek(label_offset)
        for _ in range(n_items):
            labels.append(f.read(16).decode('utf-8').strip())

        f.seek(offset_offset)
        offsets = np.fromfile(f, int, n_items)

        f.seek(len_offset)
        lens = np.fromfile(f, int, n_items)

        f.seek(maxlen_offset)
        maxlens = np.fromfile(f, int, n_items)

        f.seek(type_offset)
        types = np.fromfile(f, int, n_items)

        info = {}
        for label, offset, length, maxlen, typ in zip(labels, offsets, lens, maxlens, types):
            info[label] = (offset, length, maxlen, typ)
        self.info = info
        self.f = f
        self.info2 = {}
        self._build_info2('d', True)
        self._build_info2('i', True)
        self.scalars = self._build_scalars()
        #  Call to _build_info2 of arrays are usually not necessary
        #  runfile._build_info2('c', False)
        #  runfile._build_info2('d', False)
        #  runfile._build_info2('i', False)

    def _build_info2(self, typ: str, scalar: bool):
        assert typ in "cdi"
        if scalar:
            labels = split_str(self.get_array(typ + 'Scalar labels'))
            vals = self.get_array(typ + 'Scalar values')
            idxs = self.get_array(typ + 'Scalar indices')
            info = {}
            for label, v, idx in zip(labels, vals, idxs):
                info[label] = (v, idx)
            self.info2[(typ, scalar)] = info
        else:
            labels = split_str(self.get_array(typ + 'Array labels'))
            lens = self.get_array(typ + 'Array lengths')
            idxs = self.get_array(typ + 'Array indices')
            info = {}
            for label, l, idx in zip(labels, lens, idxs):
                info[label] = (l, idx)
            self.info2[(typ, scalar)] = info

    def _build_scalars(self):
        scalars = {}
        for (typ, scalar), i in self.info2.items():
            if not scalar:
                continue
            for label, (v, idx) in i.items():
                if idx == 0:
                    continue
                scalars[label] = v
        return scalars

    def get_array(self, key, default=None):
        try:
            offset, length, maxlen, typ = self.info[key]
        except KeyError:
            return default
        self.f.seek(offset * 512)
        if typ == 1: # int
            return np.fromfile(self.f, int, length)
        elif typ == 2: # float
            return np.fromfile(self.f, float, length)
        elif typ == 3: # str
            return self.f.read(length).decode('utf-8').strip()

    def get_scalar(self, key, default=None):
        return self.scalars.get(key, default)

    def __getitem__(self, key):
        v = self.get_array(key)
        if v is not None:
            return v
        v = self.get_scalar(key)
        if v is not None:
            return v
        raise KeyError(f'Label {key} not found in RunFile')

    def __contains__(self, key):
        return key in self.info or key in self.scalars

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.f.close()

if __name__ == '__main__':
    import sys

    labels = sys.argv[2:]
    with runfile_reader(sys.argv[1]) as runfile:
        if labels:
            for label in labels:
                print(label)
                print(runfile[label])
                print()
        else:
            type_map = [None, 'int', 'float', 'str']
            for label, (offset, length, maxlen, typ) in runfile.info.items():
                print(f'{length:5d} / {maxlen:5d}', type_map[typ], repr(label), sep='\t')
            for label, v in runfile.scalars.items():
                print(f'{repr(label):20}{v}')

