from tendo import singleton
me = singleton.SingleInstance()

import win32pipe, win32file
import sys
import time
import numpy as np
from Exploration import Predictor
import threading
import queue
# FIXED, TRY SIMPLE BACKTESTING WITH 2 LAYER RNN

class PipeServer:
    def __init__(self, pipe_name, file_name, number_of_symbols):
        self.model = None
        self.pipe_name = pipe_name
        # For multiple trading at the same time
        self.number_of_symbols = int(number_of_symbols)
        self.pipe_handles = self.create_named_pipes()
        # For data input
        self.pipe_handle = self.create_named_pipe()
        self.file_name = str(file_name)
        self.list_of_inputs = []
        self.list_of_targets = []
        self.list_of_secondary_inputs = []

    def wait_for_pipe_data(self, handle_index, pipe_q):
        is_connected = win32pipe.ConnectNamedPipe(self.pipe_handles[handle_index], None)
        if is_connected is False:
            print(is_connected)
        else:
            data = win32file.ReadFile(self.pipe_handles[handle_index], 65536)
            pipe_q.put((data, handle_index))

    def create_named_pipe(self):
        return win32pipe.CreateNamedPipe(self.pipe_name,
                                      win32pipe.PIPE_ACCESS_DUPLEX,
                                      win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
                                      255, 65536, 65536, 300, None)

    def create_named_pipes(self):
        named_pipes = []
        for i in range(self.number_of_symbols):
            named_pipes.append(win32pipe.CreateNamedPipe(self.pipe_name + str(i),
                                      win32pipe.PIPE_ACCESS_DUPLEX,
                                      win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
                                      255, 65536, 65536, 300, None))
        return named_pipes


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
                        np.save("C:/Users/louis/Documents/GitHub/DeepLearningMT4Bridge/" + self.file_name,
                                np.array(self.list_of_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/DeepLearningMT4Bridge/sec_" + self.file_name,
                                np.array(self.list_of_secondary_inputs))
                        np.save("C:/Users/louis/Documents/GitHub/DeepLearningMT4Bridge/targets_" + self.file_name,
                                np.array(self.list_of_targets))
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
                time.sleep(1000)

    def start_prediction_loop(self):
        pred = Predictor(filename=self.file_name + "_pred")

        data_q = queue.Queue()
        for handle_index in range(self.number_of_symbols):
            t = threading.Thread(target=self.wait_for_pipe_data, args=(handle_index, data_q), daemon=True)
            t.start()

        keep_looping = True
        while keep_looping:
            try:
                data, symbol_index = data_q.get()
                if data[1].decode('utf-16') == 'Done':
                    pass
                    # keep_looping = False
                else:
                    self.list_of_inputs = []
                    self.list_of_secondary_inputs = []
                    data = str(data[1], 'utf-16').split('|')
                    # print(data)
                    self.list_of_secondary_inputs.append(list(map(np.float32, data[0].split(','))))
                    new_input = []
                    for i in range(1, len(data) -1):
                        new_input.append(list(map(np.float32, data[i].split(','))))
                    self.list_of_inputs.append(new_input)
                    np.save("C:/Users/louis/Documents/GitHub/DeepLearningMT4Bridge/" + self.file_name + "_pred",
                            np.array(self.list_of_inputs))
                    np.save("C:/Users/louis/Documents/GitHub/DeepLearningMT4Bridge/sec_" + self.file_name + "_pred",
                            np.array(self.list_of_secondary_inputs))
                    prediction = pred.predict()
                    if (np.argmax(prediction[0][0]) == 2) & (np.argmax(prediction[1][0]) == 2) & (
                        np.argmax(prediction[2][0]) == 2) & (np.argmax(prediction[3][0]) == 2) & (
                        np.argmax(prediction[4][0]) == 2):
                        win32file.WriteFile(self.pipe_handles[symbol_index], b'Buy')
                    elif (np.argmax(prediction[0][0]) == 1) & (np.argmax(prediction[1][0]) == 1) & (
                    np.argmax(prediction[2][0]) == 1) & (np.argmax(prediction[3][0]) == 1) & (
                         np.argmax(prediction[4][0]) == 1):
                        win32file.WriteFile(self.pipe_handles[symbol_index], b'Sell')
                    else:
                        win32file.WriteFile(self.pipe_handles[symbol_index], b'Pass')

                win32file.FlushFileBuffers(self.pipe_handles[symbol_index])
                win32pipe.DisconnectNamedPipe(self.pipe_handles[symbol_index])
                t = threading.Thread(target=self.wait_for_pipe_data, args=(symbol_index, data_q), daemon=True)
                t.start()
            except Exception as e:
                print(e)
                time.sleep(1000)

if __name__ == '__main__':

    suffix, timeframe, number_of_symbols, type_of_datafeed = sys.argv[1:]
    print(suffix, timeframe, number_of_symbols, type_of_datafeed)
    p = PipeServer(r'\\.\pipe\\MT4_Train', str(suffix) + str(timeframe), number_of_symbols)

    if type_of_datafeed == 'Train':
        p.start_input_output_listening_loop()
    elif type_of_datafeed == 'LiveTrade':
        p.start_prediction_loop()
