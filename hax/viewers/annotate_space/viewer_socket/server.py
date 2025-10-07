#!/usr/bin/env python
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreos@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import sys
import os
import shutil
import numpy as np
import socket
from contextlib import closing
import pickle
import struct
import importlib.util


def load_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Server:

    def __init__(self, mode, metadata, port=None, verbose=False):
        self.host = socket.gethostname()
        self.port = port if port is not None else self.getFreePort()
        self.verbose = verbose
        self.metadata = metadata
        self.mode = mode

        # Socket initialization
        self.createSocket()
        self.bindSocket()

        # Prepare map generation
        self.prepareMapGeneration()

        # Listen to client
        self.addListener()

    @classmethod
    def getFreePort(cls):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        return port

    def createSocket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def bindSocket(self):
        self.server_socket.bind((self.host, self.port))

    def addListener(self):
        # threading.Thread(target=self.listenToClient).start()
        self.listenToClient()
        if self.verbose:
            print(f"Server listening on {self.host}:{self.port}")

    def listenToClient(self):
        self.server_socket.listen(1)
        self.allowConnection()
        while True:
            try:
                raw_msglen = self.recMsg(4)
                # data = self.client_socket.recv(4096, socket.MSG_DONTWAIT | viewer_socket.MSG_PEEK)
                if raw_msglen:
                    self.generateMap(raw_msglen)
            except ConnectionResetError:
                return False

    def allowConnection(self):
        self.client_socket, addr = self.server_socket.accept()
        if self.verbose:
            print(f"Got a connection from {addr}")

    def prepareMapGeneration(self):
        self.outPath = os.path.join(self.metadata["outdir"], "decoded_map_class_{:02d}.mrc")

        # Load model
        if "server_functions_path" in self.metadata and self.mode != "FromFiles":
            self.loaded_module = load_module_from_path(self.metadata["server_functions_path"], "loaded_module")

            # This function always takes kwargs as only input, and uses them to prepare a given program to decode states
            self.heterogeneity_program_interface = self.loaded_module.HeterogeneityProgramInterface(_path_template=self.outPath,
                                                                                                    _program_loading_params=self.metadata)

    def generateMap(self, raw_msglen):
        msglen = struct.unpack('>I', raw_msglen)[0]
        z_file = self.recMsg(msglen)
        z_file = pickle.loads(z_file)

        if not self.mode == "FromFiles":
            z = np.loadtxt(z_file)
            z = z[None, ...] if z.ndim == 1 else z

            self.heterogeneity_program_interface.decode_state_from_latent(latent=z)
        else:
            with open(z_file, 'r') as f:
                volumesPaths = f.read().splitlines()

            idx = 0
            for f in volumesPaths:
                shutil.copyfile(f, self.outPath.format(idx + 1))
                idx += 1

        self.client_socket.sendall("Map generated".encode())

    def recMsg(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.client_socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def closeConnection(self):
        self.client_socket.close()
        self.server_socket.close()


def main():
    import argparse

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_file', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)

    args = parser.parse_args()

    with open(args.metadata_file, 'rb') as fp:
        metadata = pickle.load(fp)

    server = Server(args.mode, metadata, port=args.port)


if __name__ == '__main__':
    import re
    import sys

    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
