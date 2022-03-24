import soundfile as sf
import numpy as np
from scipy import fftpack
from scipy import signal
from matplotlib import pyplot as plt
import csv
import pylab
import glob

count = 0
PdftLog_file = open('test_PdftLog.txt','w')         #元波形
test_lifspe_raw = open('test_lifspe_raw.txt','w')   #リフタリング後生波形
lifspe_peek = open('lifspe_peek.txt','w')    #ケプストラム書き出しファイル作成
liftering_file = open('test_liftering.txt','w')     #リフタリング前波形
sub_lifspe_file = open('sub_liftering.txt','w')     #リフタリング後平均引き算波形
cps_before_liftering_file= open('test_cps_before_liftering.txt','w')
#cps_after_liftering_file= open('test_cps_after_liftering.txt','w')
SVMcsvfile = open('SVMcsvfile.txt','w')

def hanwindow(windowlength):    #ハニング窓の作成関数
    window = np.hanning(int(windowlength))

    return window

def read_wav(input_file):   #wavファイル開く関数
    wavdata, samplerate = sf.read(input_file) # wavファイルを開き、波形とサンプリングレートを取得

    return wavdata, samplerate


def calculation_F0(n, high, quefrency, cutwav, samplerate, input, cepCoef, cutF0_high, cutF0_low): #F0の計算関数
    # FFT(スペクトル)
    global count
    #print(count)
    #print(cutwav)
    spec = np.fft.fft(cutwav,n) # x+yi
    #print(spec)
    # 振幅スペクトル
    Adft = np.abs(spec) # 虚数部をなくしている
    # パワースペクトル
    Pdft = np.abs(spec) ** 2

    # 対数振幅スペクトル
    AdftLog = 20 * np.log10(Adft)
    # 対数パワースペクトル
    PdftLog = 10 * np.log10(Pdft)

    for i in range(len(PdftLog)):
        PdftLog_file.write(str(PdftLog[i].real))                          # テキストファイルに書き込み
        PdftLog_file.write('\n')

    # 逆FFT(ケプストラム)
    cps = np.real(np.fft.ifft(PdftLog))
    cpsLif = np.array(cps)   # arrayをコピー

    for i in range(1000):
        cps_before_liftering_file.write(str(cpsLif[i].real))                          # テキストファイルに書き込み
        cps_before_liftering_file.write('\n')
        #print(cpsLif[i].real)

    # 高周波成分を除く（左右対称なので注意）
    cpsLif[cepCoef:len(cpsLif) - cepCoef + 1] = 0
    # ケプストラム領域をフーリエ変換してスペクトル領域に戻す
    # リフタリング後の対数スペクトル
#    for i in range(len(cpsLif)):
#        cps_after_liftering_file.write(str(cpsLif[i].real))                          # テキストファイルに書き込み
#        cps_after_liftering_file.write('\n')
#        print(cpsLif[i].real)

    dftSpc = np.fft.fft(cpsLif, n)
    for i in range(len(dftSpc)):
        liftering_file.write(str(dftSpc[i].real))                          # テキストファイルに書き込み
        liftering_file.write('\n')
#        print(dftSpc[i].real)

    # 元の対数スペクトルからリフタリング後の対数スペクトルを引く
    lifspe = []
    sub_lifspe_peek = []

    ###各弦のピークを抽出するにはbase_freqを変更してください(A=440, D=294)
    base_freq=294          #D=294
    #base_freq=440

    A_stoploop=10   #ピーク数20
    sum = 0

    #リフタリング波形を元波形から引いています
    for i in range(len(dftSpc)):
        if PdftLog[i] > dftSpc[i].real:
            lifspe.append(PdftLog[i]-dftSpc[i].real)

        else:
            lifspe.append(dftSpc[i].real-PdftLog[i])

    #引いた結果を書き出ています
    for i in range(len(lifspe)):
        #生データ書き出し
        test_lifspe_raw.write(str(lifspe[i].real))                          # テキストファイルに書き込み
        test_lifspe_raw.write('\n')

    #リフタリング後波形を平均で引いています
    for i in range(len(lifspe)):
        #lifspe_sub
        sum += lifspe[i]
    ave = sum / len(lifspe)

    for k in range(len(lifspe)):
        sub_lifspe_peek.append(abs(lifspe[k]) - abs(ave))
        sub_lifspe_file.write(str(sub_lifspe_peek[k]))
        sub_lifspe_file.write('\n')

    #ピーク抽出整理（base_freqの前後30Hzまでの範囲の最大値をbase_freqに代入）
    for x in range(A_stoploop):
        #下20個～
        for re_pm in range(30,0,-1):
            if(lifspe[base_freq*x]<lifspe[base_freq*x-re_pm]):
                lifspe[base_freq*x]=lifspe[base_freq*x-re_pm]
        #～上20個
        for pm in range(1,30):
            if(lifspe[base_freq*x]<lifspe[base_freq*x+pm]):
                lifspe[base_freq*x]=lifspe[base_freq*x+pm]
        lifspe_peek.write(str(lifspe[base_freq*x].real))
        lifspe_peek.write('\n')
#        if x==19:
#            lifspe_peek.write('------------------------\n')
        print(lifspe[x])
#yumi
    for i in range(1,11):
        if i==1:
            SVMcsvfile.write('2 ')
        SVMcsvfile.write(str(i))
        SVMcsvfile.write(':')
        SVMcsvfile.write(str(adrate*lifspe[base_freq*i].real))                          # テキストファイルに書き込み
        SVMcsvfile.write(' ')
    SVMcsvfile.write('\n')



def main(path, n, high, cutlength, shiftlength, cutF0_high, cutF0_low, output_sd_name, cepCoef):
    global count
    # フォルダ内ファイルの読み込み
    input_list = glob.glob(path)

    # 標準偏差を格納する配列の準備
    F0sd_list = []
    SPsd_list = []

    # # 標準偏差のリストの出力用
    # f = open(output_sd_name, "w")

    for input_file in input_list:
        f = open(input_file + output_sd_name, 'w')

        count = 0
        print(input_file)
        #f.write(input_file + ',')
        # 音データの読み込み
        wavdata, samplerate = read_wav(input_file) # wavファイルを開き、波形とサンプリングレートを取得

        t = np.arange(0.0, len(wavdata) / samplerate, 1 / samplerate)   #F0求めるときに使う

        # 切り出し準備
        windowcutlength = samplerate * cutlength   # 切り出す長さ[点]
        windowshiftlength = samplerate * shiftlength    # 窓をずらす長さ[点]
        shiftparcent = shiftlength / cutlength  # 切り出す長さに対するずらす割合

        loopnum = len(wavdata) - windowcutlength * (1.0 - shiftparcent)
        loopnum = loopnum / (windowcutlength * shiftparcent)   # 窓の数(何個F0とSPが出力されるか)

        # 窓関数準備
        hanningWindow = hanwindow(windowcutlength)

        # F0とSPを格納する配列を準備
        F0_list = []
        SP_list = []

        #窓解析を20回分
        for num in range(100):      #int(loopnum)
            a = num * windowshiftlength #aから
            b = a + windowcutlength #bまで切り出す
            cutwav = wavdata[int(a):int(b)]# 窓幅で切り出し

            # 窓処理
            cutwav = cutwav * hanningWindow
            # F0の計算
            time = t[int(a):int(b)]
            quefrency = time - min(time)
            F0 = calculation_F0(n, high, quefrency, cutwav, samplerate, input_file, cepCoef, cutF0_high, cutF0_low)


if __name__ == '__main__':
    ######## 変数の定義部分 #########################
    # main内の要変更箇所も確認する
    #path = "C:/Users/TOMOKI/Desktop/violin_kiritori_2s/D_ookikunaname_hajime2.7s_2.wav" # フォルダの指定
    path = "C:/Users/kinak/Downloads/切り取り7s音源/切り取り7s音源/D_sukoshinaname1_7s.wav"  # フォルダの指定
    #path = "C:/Users/TOMOKI/Desktop/violin_kiritori_2s/D_ookikunaname_hajime2.7s_2.wav" # フォルダの指定
    output_sd_name = "f0sp.csv"  #標準偏差の出力ファイルの名前
    # path = "./fig/*.wav"  # フォルダの指定
    # output_sd_name = "./fig/test_0707.csv"  #標準偏差の出力ファイルの名前
    # noise_path = "eventhall_noise_10n2.wav"  # 足す雑音のファイルを指定
    n = 44100 #FFTの点数
    high = 40   #何次以上を高次とするか(未使用)
    cutlength = 0.4    # 切り出す長さ[s]　要考察
    shiftlength = 0.07  # 窓をずらす長さ[s] 要考察
    cutF0_high = 300.0       # これ以上の周波数は抽出誤りとする[Hz](未使用)
    cutF0_low = 50.0         # これ以下の周波数は抽出誤りとする[Hz](未使用)
    cepCoef = 40             # ケプストラム次数でどこまで残すか決める 元40
    #################################################

    main(path, n, high, cutlength, shiftlength, cutF0_high, cutF0_low, output_sd_name, cepCoef)

PdftLog_file.close()
test_lifspe_raw.close()
liftering_file.close()
lifspe_peek.close()
sub_lifspe_file.close()
SVMcsvfile.close()
cps_before_liftering_file.close()
#cps_after_liftering_file.close()
