import numpy as np

class DataDealer:
    def __init__(self, suffix, timeframe):
        self.suffix = suffix
        self.timeframe = timeframe
        self.target_df = np.load('targets_' + str(suffix) + str(timeframe) + '.npy')
        self.attribute_df = np.load(str(suffix) + str(timeframe) + '.npy')
        self.type_df = np.load('sec_' + str(suffix) + str(timeframe) + '.npy')
        self.type_size = 0
        self.data_length = 0
        self.data_types = 0

    def double_input_single_multiple_output_split(self, categorical=True):

        # One hot encoding
        hours_df = np.zeros(shape=(self.type_df.shape[0], 24))
        for i, hours in enumerate(self.type_df[:,1]):
            hours_df[i][int(hours)] = 1
        type_df = np.zeros(shape=(self.type_df.shape[0], int(self.type_df[:,0].max() + 1)))
        for i, type in enumerate(self.type_df[:,0]):
            type_df[i][int(type)] = 1

        self.type_df = np.hstack((type_df, hours_df))
        self.type_size = self.type_df.shape[-1]
        mean_df = self.attribute_df.mean(axis=1)

        self.target_df[:, 0] = (self.target_df[:, 0] - self.attribute_df[:, -1, 0]) / mean_df[:, 0]
        self.target_df[:, 1] = (self.target_df[:, 1] - self.attribute_df[:, -1, 0]) / mean_df[:, 0]
        self.target_df[:, 2] = (self.target_df[:, 2] - self.attribute_df[:, -1, 0]) / mean_df[:, 0]
        self.target_df[:, 3] -= self.target_df[:, 3].mean()
        self.target_df[:, 3] /= self.target_df[:, 3].std()
        if categorical is True:
            self.target_df[:,0] = (self.target_df[:,0] > 0).astype(int)
        else:
            self.target_df[:,0] /= self.target_df[:,0].std()
        self.target_df[:, 1] /= self.target_df[:, 1].std()
        self.target_df[:, 2] /= self.target_df[:, 2].std()

        self.attribute_df[:,:-1,0] = (self.attribute_df[:,:-1,0] - self.attribute_df[:,1:,0]) / mean_df[:,None,0]
        self.attribute_df = self.attribute_df[:,:-1,]
        self.attribute_df[:,:,:2] /= mean_df[:,None,:2]
        self.attribute_df[:, :, 0] /= self.attribute_df[:, :, 0].std()
        self.attribute_df[:, :, 1] /= self.attribute_df[:, :, 1].std()
        self.attribute_df[:,:,2] -= self.attribute_df[:,:,2].mean()
        self.attribute_df[:,:,2] /= self.attribute_df[:,:,2].std()
        self.data_types = self.attribute_df.shape[-1]
        self.data_length = self.attribute_df.shape[1]

if __name__ == '__main__':
    data = DataDealer('forex', 30)
    data.double_input_single_multiple_output_split()