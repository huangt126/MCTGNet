import logging
import os
import numpy as np
import pandas as pd
import random
import datetime
import time

from pandas import ExcelWriter
from torch.autograd import Variable
from torchsummary import summary
import torch
from torch.backends import cudnn
from torch.utils.data import TensorDataset, DataLoader
from utils import calMetrics, calculatePerClass, numberClassChannel, load_data_evaluate
from model import MCTGNet
import torch.nn.functional as F
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.benchmark = False
cudnn.deterministic = True

class ExP():
    def __init__(self, nsub, data_dir, result_name,
                 epochs=2000,
                 number_aug=2,
                 number_seg=8,
                 gpus=[0],
                 evaluate_mode='subject-dependent',
                 heads=4,
                 emb_size=40,
                 depth=6,
                 dataset_type='B',
                 eeg1_f1=8,
                 eeg1_kernel_sizes=(32, 64, 96),
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3,
                 flatten_eeg1=600,
                 learning_rate=0.0005,
                 batch_size=72,
                 current_time=None,
                 ):


        super(ExP, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.cr_tag = 0
        self.n_epochs = epochs
        self.nSub = nsub
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.root = data_dir
        self.heads = heads
        self.emb_size = emb_size
        self.depth = depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        # 构建结果目录路径
        self.current_time = current_time
        # 构建结果目录路径
        params_str = f"{eeg1_kernel_sizes}{self.dataset_type}_{self.heads}_{self.depth}"
        # 构建结果目录路径
        self.results_dir = os.path.join("/root/autodl-tmp/mctgnet_result", f"{params_str}_{self.current_time}")

        # 创建目录（包括父目录）
        os.makedirs(self.results_dir, exist_ok=True)
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logging.basicConfig(filename=f'{self.results_dir}/log_{nsub}_{now_time}.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.number_class, self.number_channel = numberClassChannel(self.dataset_type)
        self.model = MCTGNet(
            heads=self.heads,
            emb_size=self.emb_size,
            depth=self.depth,
            database_type=self.dataset_type,
            eeg1_f1=eeg1_f1,
            eeg1_D=eeg1_D,
            eeg1_kernel_sizes=eeg1_kernel_sizes,
            eeg1_pooling_size1=eeg1_pooling_size1,
            eeg1_pooling_size2=eeg1_pooling_size2,
            eeg1_dropout_rate=eeg1_dropout_rate,
            eeg1_number_channel=self.number_channel,
            flatten_eeg1=flatten_eeg1,
        ).cuda()
        # self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.model = self.model.cuda()
        self.model_filename = self.results_dir + '/model_{}.pth'.format(self.nSub)

    # Segmentation and Reconstruction (S&R)数据增强
    def interaug(self, timg, label):
        """
        对输入数据进行增强，生成用于训练的增强数据和对应的标签。

        参数:
        timg: 输入的数据，形状为(batch_size, 1, number_channel, 1000)。
        label: 输入的数据，形状为(batch_size,)。

        返回:
        aug_data: 增强后的数据，转换为PyTorch张量并移动到GPU上。
        aug_label: 增强后的标签数据，转换为PyTorch张量并移动到GPU上。
        """
        # 初始化增强数据和标签的列表
        aug_data = []
        aug_label = []

        # 计算每种增强方法生成的记录数
        number_records_by_augmentation = self.number_augmentation * int(self.batch_size / self.number_class)
        # 计算每个分段的点数
        number_segmentation_points = 1000 // self.number_seg

        # 遍历每个类别进行数据增强
        for clsAug in range(self.number_class):
            # 找到当前类别的索引
            cls_idx = np.where(label == clsAug + 1)
            # 提取当前类别的数据和标签
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            # 初始化当前类别增强后的数据数组
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            # 对每个记录进行增强
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    # 随机选择样本索引进行拼接
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    # 将选中的样本片段复制到增强数据中
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :,
                        rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            # 将当前类别的增强数据和标签添加到列表中
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:number_records_by_augmentation])

        # 将列表中的数据合并为一个数组
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)

        # 生成增强数据的随机索引以打乱数据
        aug_shuffle = np.random.permutation(len(aug_data))
        # 根据随机索引打乱增强数据和标签
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        # 将增强数据和标签转换为PyTorch张量并移动到GPU上
        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()

        # 返回增强后的数据和标签
        return aug_data, aug_label


    def get_source_data(self):
        (self.train_data,  # (batch, channel, length)
         self.train_label,
         self.test_data,
         self.test_label) = load_data_evaluate(self.root, self.dataset_type, self.nSub,
                                               mode_evaluate=self.evaluate_mode)
        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)
        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.test_data - target_mean) / target_std
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img).to(self.device)
        label = torch.from_numpy(label - 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data).to(self.device)
        test_label = torch.from_numpy(test_label - 1).to(self.device)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                           shuffle=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        best_test_acc = 0
        best_epoch = 0
        result_process = []

        for e in range(self.n_epochs):
            epoch_process = {}
            epoch_process['epoch'] = e
            self.model.train()
            outputs_list = []
            label_list = []

            for i, (img_batch, label_batch) in enumerate(self.dataloader):

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)

                aug_data = aug_data.to(self.device)
                aug_label = aug_label.to(self.device)

                img_batch = torch.cat((img_batch, aug_data))
                label_batch = torch.cat((label_batch, aug_label))

                img_batch = img_batch.type(self.Tensor)
                label_batch = label_batch.type(self.LongTensor)

                _, _,features, outputs = self.model(img_batch)
                outputs_list.append(outputs)
                label_list.append(label_batch)

                loss = self.criterion_cls(outputs, label_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_pred = torch.max(torch.cat(outputs_list), 1)[1].cpu()
                train_label = torch.cat(label_list).cpu()
                train_acc = float((train_pred == train_label).numpy().astype(int).sum()) / float(train_label.size(0))
                epoch_process['train_acc'] = train_acc

            del img_batch, label_batch, aug_data, aug_label
            torch.cuda.empty_cache()

            self.model.eval()
            test_loss = 0
            outputs_list = []

            with torch.no_grad():
                for batch_idx, (img, _) in enumerate(self.test_dataloader):
                    img = img.type(self.Tensor)
                    _, _,_,cls_outputs = self.model(img)
                    outputs_list.append(cls_outputs)

                    test_loss += self.criterion_cls(
                        cls_outputs.float(),
                        test_label[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size].long()
                    ).item()

                    del img, cls_outputs
                    torch.cuda.empty_cache()

            cls_outputs = torch.cat(outputs_list)
            test_pred = torch.max(cls_outputs, 1)[1]
            test_acc = float((test_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

            test_loss /= len(self.test_dataloader)

            epoch_process['test_acc'] = test_acc
            epoch_process['train_loss'] = loss.detach().cpu().numpy()
            epoch_process['test_loss'] = test_loss

            log_message = f"Subject {self.nSub}: Epoch {e:>3}: Train Loss = {loss.detach().cpu().numpy():.4f}, Train Acc = {train_acc:.4f}, Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}"
            print(log_message)
            logging.info(log_message)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = e
                torch.save({
                    'epoch': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_accuracy': test_acc,
                }, self.model_filename)

            result_process.append(epoch_process)

            del outputs_list, label_list, cls_outputs, test_pred
            torch.cuda.empty_cache()

        checkpoint = torch.load(self.model_filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        features_list = []
        outputs_features = []
        outputs_list = []
        org_list = []
        cnn_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_dataloader):
                img_test = img.type(self.Tensor)
                org, cnn, features, outputs = self.model(img_test)
                org_list.append(org.cpu().numpy())
                cnn_list.append(cnn.cpu().numpy())
                features_list.append(features.cpu().numpy())
                outputs_features.append(outputs.cpu().numpy())
                outputs_list.append(outputs.cpu())

        org_all = np.vstack(org_list)
        cnn_all = np.vstack(cnn_list)
        features_all = np.vstack(features_list)
        outputs_features_all = np.vstack(outputs_features)
        outputs = torch.cat(outputs_list)
        y_pred = torch.max(outputs, 1)[1].cpu().numpy()
        y_true = test_label.cpu().numpy()

        print(f"Subject {self.nSub}: epoch: {best_epoch}\tThe best test accuracy is: {best_test_acc}")
        logging.info(f"Subject {self.nSub}: epoch: {best_epoch}\tThe best test accuracy is: {best_test_acc}")
        df_process = pd.DataFrame(result_process)

        df_process.to_csv(os.path.join(self.results_dir, f'training_process_subject_{self.nSub}.csv'), index=False)

        np.save(os.path.join(self.results_dir, f'org_subject_{self.nSub}.npy'), org_all)
        np.save(os.path.join(self.results_dir, f'cnn_subject_{self.nSub}.npy'), cnn_all)
        np.save(os.path.join(self.results_dir, f'features_subject_{self.nSub}.npy'), features_all)
        np.save(os.path.join(self.results_dir, f'outputs_subject_{self.nSub}.npy'), outputs_features_all)
        np.save(os.path.join(self.results_dir, f'predictions_subject_{self.nSub}.npy'), y_pred)
        np.save(os.path.join(self.results_dir, f'true_labels_subject_{self.nSub}.npy'), y_true)

        with open(os.path.join(self.results_dir, f'best_accuracy_subject_{self.nSub}.txt'), 'w') as f:
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Test Accuracy: {best_test_acc:.4f}\n")

        return best_test_acc, test_label, y_pred, df_process, best_epoch


def main(dirs,
         evaluate_mode='subject-dependent',  # "LOSO" or other
         heads=3,  # heads of MHA
         emb_size=16,  # token embding dim
         depth=6,  # Transformer encoder depth
         dataset_type='A',  # A->'BCI IV2a', B->'BCI IV2b'
         eeg1_f1=8,  # features of temporal conv
         eeg1_kernel_sizes=(32, 64, 96),  # kernel size of temporal conv
         eeg1_D=2,  # depth-wise conv
         eeg1_pooling_size1=8,  # p1
         eeg1_pooling_size2=8,  # p2
         eeg1_dropout_rate=0.3,
         flatten_eeg1=240,
         ):
    print("*" * 40)
    # print(f"eeg1_kernel_sizes is :{eeg1_kernel_sizes}")
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # result_write_metric = ExcelWriter(dirs + "/result_metric.xlsx")

    result_metric_dict = {}
    y_true_pred_dict = {}

    # process_write = ExcelWriter(dirs + "/process_train.xlsx")
    # pred_true_write = ExcelWriter(dirs + "/pred_true.xlsx")
    subjects_result = []
    best_epochs = []
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(0, 54):
    # for i in [1,3,4,5]:
        starttime = datetime.datetime.now()
        np.random.seed(2025530)
        seed_n = np.random.randint(2025530)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        index_round = 0
        print('Subject %d' % (i + 1))
        exp = ExP(i + 1, DATA_DIR, dirs, EPOCHS, N_AUG, N_SEG, gpus,
                  evaluate_mode=evaluate_mode,
                  heads=heads,
                  emb_size=emb_size,
                  depth=depth,
                  dataset_type=dataset_type,
                  eeg1_f1=eeg1_f1,
                  eeg1_kernel_sizes=eeg1_kernel_sizes,
                  eeg1_D=eeg1_D,
                  eeg1_pooling_size1=eeg1_pooling_size1,
                  eeg1_pooling_size2=eeg1_pooling_size2,
                  eeg1_dropout_rate=eeg1_dropout_rate,
                  flatten_eeg1=flatten_eeg1,
                  current_time=current_time,
                  )

        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int) if hasattr(Y_true, 'cpu') else Y_true.astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int) if hasattr(Y_pred, 'cpu') else Y_pred.astype(int)
        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        y_true_pred_dict[i] = df_pred_true

        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {'accuray': accuracy * 100,
                          'precision': precison * 100,
                          'recall': recall * 100,
                          'f1': f1 * 100,
                          'kappa': kappa * 100
                          }
        subjects_result.append(subject_result)
        best_epochs.append(best_epoch)

        print('Subject %d: THE BEST ACCURACY IS ' % (i + 1) + str(testAcc) + "\tkappa is " + str(kappa))

        endtime = datetime.datetime.now()
        print('Subject %d: duration: ' % (i + 1) + str(endtime - starttime))
        try:
            if isinstance(Y_true, np.ndarray):
                Y_true = torch.from_numpy(Y_true)
            if isinstance(Y_pred, np.ndarray):
                Y_pred = torch.from_numpy(Y_pred)
            if i == 0:
                yt = Y_true
                yp = Y_pred
            else:
                # continue
                yt = torch.cat((yt, Y_true))
                yp = torch.cat((yp, Y_pred))

            df_result = pd.DataFrame(subjects_result)
        finally:
            pass
    try:
        print('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) + "kappa is: " + str(
            df_result['kappa'].mean()) + "\n")
        logging.info('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) + "kappa is: " + str(
            df_result['kappa'].mean()) + "\n")
        print("best epochs: ", best_epochs)
        logging.info("f'best epochs: ', {best_epochs}")
        # df_result.to_excel(result_write_metric, index=False)
        result_metric_dict = df_result

        mean = df_result.mean(axis=0)
        mean.name = 'mean'
        std = df_result.std(axis=0)
        std.name = 'std'
        df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])

        # df_result.to_excel(result_write_metric, index=False)
        print('-' * 9, ' all result ', '-' * 9)
        logging.info("'-' * 9, ' all result ', '-' * 9")
        print(df_result)
        logging.info(df_result)

        print("*" * 40)
        logging.info("*" * 40)

        # result_write_metric.close()
    finally:
        pass

    return result_metric_dict


if __name__ == "__main__":
    # ----------------------------------------
        DATA_DIR = r'/root/'
        EVALUATE_MODE = 'LOSO-No'  # leaving one subject out subject-dependent  subject-indenpedent

        N_SUBJECT = 9  # BCI
        N_AUG = 2  # 数据增强次数用于生成人工训练数据集
        N_SEG = 8  # S&R的分割次数

        EPOCHS = 2000
        EMB_DIM = 16
        HEADS = 4
        DEPTH = 1
        TYPE = 'A'

        EEGNet1_F1 = 8
        EEGNet1_KERNEL_SIZES = (32,64,96)
        EEGNet1_D = 2
        EEGNet1_POOL_SIZE1 = 8
        EEGNet1_POOL_SIZE2 = 8
        FLATTEN_EEGNet1 = 240
        if EVALUATE_MODE != 'LOSO':
            EEGNet1_DROPOUT_RATE = 0.5
        else:
            EEGNet1_DROPOUT_RATE = 0.25
        number_class, number_channel = numberClassChannel(TYPE)
        RESULT_NAME = "{}heads{}depth{}".format(TYPE, HEADS, DEPTH)
        sModel = MCTGNet(
            heads=HEADS,
            emb_size=EMB_DIM,
            depth=DEPTH,
            database_type=TYPE,
            eeg1_f1=EEGNet1_F1,
            eeg1_D=EEGNet1_D,
            eeg1_kernel_sizes=EEGNet1_KERNEL_SIZES,
            eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
            eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
            eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
            eeg1_number_channel=number_channel,
            flatten_eeg1=FLATTEN_EEGNet1,
        ).cuda()
        summary(sModel, (1, number_channel, 1000))
        print(time.asctime(time.localtime(time.time())))
        result = main(RESULT_NAME,
                      evaluate_mode=EVALUATE_MODE,
                      heads=HEADS,
                      emb_size=EMB_DIM,
                      depth=DEPTH,
                      dataset_type=TYPE,
                      eeg1_f1=EEGNet1_F1,
                      eeg1_kernel_sizes=EEGNet1_KERNEL_SIZES,
                      eeg1_D=EEGNet1_D,
                      eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
                      eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
                      eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
                      flatten_eeg1=FLATTEN_EEGNet1,
                      )
        print(time.asctime(time.localtime(time.time())))
