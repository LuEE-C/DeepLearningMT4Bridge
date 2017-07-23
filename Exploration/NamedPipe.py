import win32pipe, win32file
import sys
import os
from io import StringIO
import pandas as pd
import time
import numpy as np
from Exploration.Exploration import train, predict

# FIXED, TRY SIMPLE BACKTESTING WITH 2 LAYER RNN

class PipeServer:
    def __init__(self, pipe_name, file_name, number_of_symbols):
        self.model = None
        self.pipe_name = pipe_name
        self.pipe_handle = self.create_named_pipe()
        self.file_name = str(file_name)
        self.number_of_symbols = int(number_of_symbols)
        self.list_of_inputs = []
        self.list_of_targets = []
        self.list_of_secondary_inputs = []

    def create_named_pipe(self):
        return win32pipe.CreateNamedPipe(self.pipe_name,
                                      win32pipe.PIPE_ACCESS_DUPLEX,
                                      win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
                                      255, 65536, 65536, 300, None)

    def start_input_output_listening_loop(self):
        keep_looping = True
        while(keep_looping):
            try:
                is_connected = win32pipe.ConnectNamedPipe(self.pipe_handle, None)
                if is_connected is False:
                    print(is_connected)
                else:
                    data = win32file.ReadFile(self.pipe_handle, 65536)
                    win32file.WriteFile(self.pipe_handle, b'Nothing')
                    if data[1].decode('utf-16') == 'Done':
                        keep_looping = False
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/" + self.file_name,
                                np.array(self.list_of_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/sec_" + self.file_name,
                                np.array(self.list_of_secondary_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/targets_" + self.file_name,
                                np.array(self.list_of_targets))
                    elif data[1].decode('utf-16') == 'AllTrainingSent':
                        keep_looping = False
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/" + self.file_name,
                                np.array(self.list_of_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/sec_" + self.file_name,
                                np.array(self.list_of_secondary_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/targets_" + self.file_name,
                                np.array(self.list_of_targets))
                        train(filename=self.file_name)

                    else:
                        data = str(data[1], 'utf-16').split('|')
                        self.list_of_secondary_inputs.append(list(map(np.float32, data[0].split(','))))
                        self.list_of_targets.append(list(map(np.float32, data[1].split(','))))
                        new_input = []
                        for i in range(2, len(data) -1):
                            new_input.append(list(map(np.float32, data[i].split(','))))
                        self.list_of_inputs.append(new_input)

                win32file.FlushFileBuffers(self.pipe_handle)
                win32pipe.DisconnectNamedPipe(self.pipe_handle)
            except Exception as e:
                print(e)
                print(data)
                time.sleep(1000)

    def start_prediction_loop(self):
        keep_looping = True
        while(keep_looping):
            try:
                is_connected = win32pipe.ConnectNamedPipe(self.pipe_handle, None)
                if is_connected is False:
                    print(is_connected)
                else:
                    data = win32file.ReadFile(self.pipe_handle, 65536)
                    win32file.WriteFile(self.pipe_handle, b'Nothing')
                    if data[1].decode('utf-16') == 'Done':
                        keep_looping = False
                    else:
                        data = str(data[1], 'utf-16').split('|')
                        self.list_of_secondary_inputs.append(list(map(np.float32, data[0].split(','))))
                        new_input = []
                        for i in range(1, len(data) -1):
                            new_input.append(list(map(np.float32, data[i].split(','))))
                        self.list_of_inputs.append(new_input)
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/" + self.file_name,
                                np.array(self.list_of_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/ForexML/Exploration/sec_" + self.file_name,
                                np.array(self.list_of_secondary_inputs))
                        prediction = str(predict(filename=self.file_name))
                        win32file.WriteFile(self.pipe_handle, prediction.astype(bytes))

                win32file.FlushFileBuffers(self.pipe_handle)
                win32pipe.DisconnectNamedPipe(self.pipe_handle)

            except Exception as e:
                print(e)
                print(data)
                time.sleep(1000)

if __name__ == '__main__':
    suffix, timeframe, number_of_symbols, type_of_datafeed, bars_used, bars_type = sys.argv[1:]
    p = PipeServer(r'\\.\pipe\\MT4_Train', str(suffix) + str(timeframe), number_of_symbols, bars_used, bars_type)

    if type_of_datafeed == 'Backtest':
        p.start_input_output_listening_loop()
    elif type_of_datafeed == 'Train':
        p.start_input_output_listening_loop()
    elif type_of_datafeed == 'Livetrade':
        pass