"""
シャンテン数の計算
    あがってたらつぎへ
    国士無双,チートイツはカット
        チートイツは二盃口含みの危険性
        標準でシャンテン-1かつ七対子なら二盃口確定

index化の方法
    鳴き、字牌、マンズは同値類でまとめる
        (1,9),(2~8),(字牌)  #三元牌と風で分ける説もある

        0.トイツ マンズ
        1.トイツ 字
        2.ポン1,9
        3.ポン2~8
        4.ポン字
        5.ミンカン1,9
        6.ミンカン2~8
        7.ミンカン字
        8.アンカン1,9
        9.アンカン2~8
        10.アンカン字

        全部で(孤立チートイツ系はのぞく)
        

    数字牌部分
        if 1~ start:
            1start
        else if ~9:
            反転して1スタート
        else:
            2スタートの同値類

indexテーブルで管理するもの
    符計算!!!ほとんどこいつのため   
    平和()
    タンヤオ
    チャンタ系(鳴きも)
    七対子
    一盃口,二盃口
    トイトイ
    一通
    三暗刻
    四暗刻,単騎
    九蓮宝燈
    老頭
    槓子系

    七対子の場合のみ


プログラムで管理するもの 基本はO(1)
    国士無双
    平和の役牌チェック:sum[yakuhai]==0
    三元牌
    風役
    染め系,緑一色:sum[some]==0,some[ryu]==0
    三色同刻:1,9どちらか3つずつ抜けてシャンテン数チェック?
    ドラ、赤ドラ、裏ドラ
    門前ツモ
    リーチ、ダブリー
    一発
	チャンカン,リンシャン
	ハイテイ、ホーテイ
	流し
	天和、地和


どっちにするか迷い中    
    
    大三元、小三元
    四喜和
    

役番号
全部で52種類
0:タンヤオ    1:ピンフ     2:イーペーコ  3:対々和     4:チャンタ      5:チャンタ鳴き 6:ジュンチャン  7:ジュンチャン鳴き
8:一通        9:一通鳴き   10:混老頭     11:混一色    12:混一色鳴き   13:清一色      14:清一色鳴き   15:二盃口
16:場風 東    17:場風 南   18:場風 西    19:自風 東   20:自風 南      21:自風 西     22:白           23:発
24:中         25:三槓子    26:小三元     27:三色同刻  28:三暗刻       29:七対子      30:リーチ       31:ダブリー
32:一発       33:門前ツモ  34:嶺上開花   35:チャンカン36:海底         37:河底        38:国士         39:国士13面     
40:四暗刻     41:四暗刻単騎42:九蓮       43:純正九蓮  44:字一色       45:四槓子      46:清老頭       47:大三元       
48:小四喜     49:大四喜    50:緑一色     51:天和(地和) 

流し、ドラは別で考える？
ドラ=[赤ドラ,抜きドラ,表ドラ,裏ドラ]

(場風 北、自風 北、三色同順,地和)はcutしている

天鳳での仕様





"""
from asyncio import subprocess
from re import I
import sys
import os
import numpy as np
from operator import index
import pickle

from mahjong.hand_calculating import yaku_list
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.hand_calculating.hand_config import OptionalRules
from mahjong.meld import Meld


class my_yaku_module:
    """
    基本の手牌の入力は
        手牌29+抜き1+pon29+明カン27+暗カン27+面前1+リーチ1+一発1=116次元
    """
    def __init__(self):
        with open('shanten_three/shanten_dic.pickle','rb') as f:
            self.shanten_table=pickle.load(f)
        self.su_shanten_idxs=[[2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19]]
        self.ji_shanten_idxs=[0,1,20,21,22,23,24,25,26]
        self.terminal_idxs=[0,1,2,10,11,19]
        self.jihai_idx27=[20,21,22,23,24,25,26]
        self.jihai_str=['ton','nan','sha','pe','haku','hatsu','chun']
        self.kind_idxs=[[0,1],[2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19],[0,1,20,21,22,23,24,25,26]]
        self.yaku_length=52

        self.pai_id116to108_dic={27:6,28:15,57:33,58:42}
        self.not_ryuiso={1,2,3,4,5,6,7,8,9,10,11,15,17,19,20,21,22,23,24,26}


        #hand116部分のindex
        self.hand116_pai_sidx=0
        self.hand116_nuki_idx=29
        self.hand116_pon_sidx=30
        self.hand116_minkan_sidx=59
        self.hand116_ankan_sidx=86
        self.hand116_menzen_idx=113
        self.hand116_reach_idx=114
        self.hand116_ippatsu_idx=115
        self.hand116_akadora_idxs=[27,28,57,58,65,74,92,101]



        #yaku_dic_str=[id,han]
        self.yaku_dic_str= {'tanyao':[0,1]        ,'pinfu':[1,1]         ,'ipeko':[2,1]         ,'toitoi':[3,2]        ,'chanta':[4,2],
                            'chanta_naki':[5,1]   ,'junchan':[6,3]       ,'junchan_naki':[7,2]  ,'ittsu':[8,2]         ,'ittsu_naki':[9,1],
                            'nan_ji':[10,2]       ,'honitsu':[11,3]      ,'honitsu_naki':[12,2] ,'chinitsu':[13,6]     ,'chinitsu_naki':[14,5],
                            'ryanpeko':[15,3]     ,'ton_ba':[16,1]       ,'nan_ba':[17,1]       ,'sha_ba':[18,1]       ,'ton_ji':[19,1],
                            'nan_ji':[20,1]       ,'sha_ji':[21,1]       ,'haku':[22,1]         ,'hatsu':[23,1]        ,'chun':[24,1],
                            'sankantsu':[25,2]    ,'shosangen':[26,2]    ,'sanshoku_doko':[27,2],'sananko':[28,2]      ,'honroto':[29,2],
                            'chitoitsu':[30,2]    ,'reach':[31,1]        ,'double_reach':[32,2] ,'ippatsu':[33,1]      ,'mentsumo':[34,1],
                            'rinshan_kaiho':[35,1],'chankan':[36,1]      ,'haitei':[37,1]       ,'hotei':[38,1]        ,'kokushi':[39,13],
                            'kokushi13':[40,14]   ,'suanko':[41,13]      ,'suanko_tanki':[42,14],'churen':[43,13]      ,'churen_jun':[44,14],
                            'tsuiso':[45,13]      ,'sukantsu':[46,13]    ,'chinroto':[47,13]    ,'daisangen':[48,13]   ,'shosushi':[49,13],
                            'daisushi':[50,14]    ,'ryuiso':[51,13]      ,'tenho':[52,13]}
        self.yaku_dic_int={}
        for k,v in self.yaku_dic_str.items():
                self.yaku_dic_int[v[0]]=k


        self.yaku_length=53
        self.yakuman_set={'kokushi','kokushi13','suanko','suanko_tanki','churen','churen_jun','tsuiso','sukantsu',
                       'chinroto','daisangen','shosushi','daisushi','ryuiso','tenho'}

        


        #mj_module
        
        self.calculator_mjmodule=HandCalculator()
        self.config_mj_module=HandConfig(options=OptionalRules(has_open_tanyao=True))

        #必要な部分だけ,indexの課題
        self.yaku_dic_mjmodule={'Tanyao':0,'Pinfu':1,'Iipeiko':2,'Chanta':4,'Junchan':6,'Ittsu':8,'Toitoi':3,
                                'San Ankou':28,'San Kantsu':25,'Sanshoku Doko':27,'Suu ankou':41,'Suu ankou tanki':42,
                                'Chuuren Poutou':43,'Daburu Chuuren Poutou':44,'Suu kantsu':46}
    
    def yaku_str_list(self,yaku):
        l=[]
        for k,v in self.yaku_dic_str.items():
            if yaku[v[0]]==1:
                l.append(k)
        return l
    def transform_pai29to27(self,pai_id29):
        if 0<=pai_id29<27:
            return pai_id29
        elif 27<=pai_id29<29:
            return [6,15][pai_id29-27]
        else:
            print(f'===error:{pai_id29} input transform_pai29to27())===')
            return -1
    def next_pai27(self,pai_id27):#dora用
        next_pai_id=-1
        if 0<=pai_id27<2:
            next_pai_id=(pai_id27+1)%2
        elif 2<=pai_id27<11:
            next_pai_id=(pai_id27-2+1)%9+2
        elif 11<=pai_id27<20:
            next_pai_id=(pai_id27-11+1)%9+11
        elif 20<=pai_id27<24:
            next_pai_id=(pai_id27-20+1)%4+20
        elif 24<=pai_id27<27:
            next_pai_id=(pai_id27-24+1)%3+24
        return next_pai_id
    def transform_pai27to34(self,pai27):
        if pai27==0:
            return 0
        elif pai27<27:
            return pai27+7
        else:
            return -1
    def transform_pai116to108(self,pai116):
        if 0<=pai116<27:
            return pai116
        elif 27<=pai116<29:
            return self.transform_pai29to27(pai116)
        #nukidoraは保留
        elif 30<=pai116<57:#pon
            return pai116-3
        elif 57<=pai116<59:#rpon
            return self.transform_pai29to27(pai116-30)+27
        elif 59<=pai116<113:#minkan,ankan
            return pai116-5
        elif 0<pai116<=115:
            return -1
        else:
            return -2#error
    def transform_pai108to116(self,pai108,lis=False):
        if 0<=pai108<27:
            return pai108
        elif 27<=pai108<54:
            return pai108+3
        elif 54<=pai108<108:
            return pai108+5
        else:
            return -2#error
    def transform_pai134to37(self,pai134):#tenho style
        if pai134 == 16:
            return 34
        elif pai134 == 52:
            return 35
        elif pai134 == 88:
            return 36
        else:
            return pai134//4
    def transform_pai37to29(self,pai37):#4人 of tenho  to 3人
        if pai37==0:
            return pai37
        elif pai37<34:
            return pai37-7
        elif 35<=pai37<37:
            return pai37-8
        else:
            return -1
    def transform_hand116to108(self,hand116):
        idx=[]
        start_idx=[0,30,59,86]
        for i in range(4):
            s=start_idx[i]
            for j in range(27):
                idx.append(s+j)
        hand108=hand116[idx]
        
        #r5p,r5s:pon,
        for id116,id108 in self.pai_id116to108_dic.items():
            hand108[id108]+=hand116[id116]

        return hand108

    def transform_hand116to34(self,hand116):
        hand34=hand116[1:27]
        hand34=np.pad(hand34,(8,0))
        hand34[0]=hand116[0]
        hand34[13]+=hand116[27]
        hand34[22]+=hand116[28]
        return hand34
    def transform_hand108to34(self,hand108):
        hand34=hand108[1:27]
        hand34=np.pad(hand34,(8,0))
        hand34[0]=hand108[0]
        return hand34


    ###shanten数の計算
    def shanten(self,hand27or108,naki_mentsu=None):
        if len(hand27or108)==108:
            hand27=hand27or108[:27]
            if naki_mentsu==None:
                naki_mentsu=int(sum(hand27or108[27:]))
        elif len(hand27or108)==27:
            hand27=hand27or108
            if naki_mentsu==None:
                naki_mentsu=0
        else:
            print(f'wrong length in shanten')
        shanten_i=self.shanten_ippan(hand27,naki_mentsu=naki_mentsu)
        shanten_k=14
        shanten_c=14
        if naki_mentsu==0:
            shanten_k=self.shanten_kokushi(hand27)
            shanten_c=self.shanten_chitoi(hand27)
        return shanten_i,shanten_c,shanten_k
    def shanten_hash(self,hand9):
        s=[0,8]
        v=[1,-1]
        ans=float('inf')
        for i in range(2):
            n=0
            flag=False
            m=1
            idx=s[i]
            for j in range(9):
                if hand9[idx]:
                    if not flag:flag=True
                    n+=m*hand9[idx]
                if flag:
                    m*=5
                idx+=v[i]
            ans=min(ans,n)
        return int(ans)
    def __shanten_ippan_sub(self,ji_nl,su_nl,naki_mentsu):
        nl0=su_nl[0]
        nl1=su_nl[1]
        ma=ji_nl[0]+nl0[0]+nl1[0]
        ta=ji_nl[1]+nl0[1]+nl1[1]
        mb=ji_nl[0]+nl0[2]+nl1[2]
        tb=ji_nl[1]+nl0[3]+nl1[3]
        if ma+ta>4-naki_mentsu:
            ta=4-ma-naki_mentsu
        if mb+tb>4-naki_mentsu:
            tb=4-mb-naki_mentsu
        return min(8-2*ma-ta-naki_mentsu*2,8-2*mb-tb-naki_mentsu*2)

    def shanten_ippan(self,hand27,naki_mentsu=0):
        ji_nl=[0]*2
        for j in self.ji_shanten_idxs:
            if hand27[j]==2:
                ji_nl[1]+=1
            elif hand27[j]>2:
                ji_nl[0]+=1
        hashs=[0]*2
        for i in range(2):
            jl=self.su_shanten_idxs[i]
            key=self.shanten_hash(hand27[jl])
            hashs[i]=key
        #初期shanten数、字牌の対子があるときは抜いて考える(抜いて考えたほうが常にそれ以下になる)
        if ji_nl[1]>0:
            ji_nl[1]-=1
            min_s=self.__shanten_ippan_sub(ji_nl,[self.shanten_table[hashs[0]],self.shanten_table[hashs[1]]],naki_mentsu)-1
            ji_nl[1]+=1
        else:
            min_s=self.__shanten_ippan_sub(ji_nl,[self.shanten_table[hashs[0]],self.shanten_table[hashs[1]]],naki_mentsu)
        
        #数牌の対子を抜く
        for i in range(2):
            su_nl=[0]*2
            p=(1+i)%2
            su_nl[p]=self.shanten_table[hashs[p]]
            jl=self.su_shanten_idxs[i]
            for j in jl:
                if hand27[j]>1:
                    hand27[j]-=2
                    key=self.shanten_hash(hand27[jl])
                    hand27[j]+=2

                    su_nl[i]=self.shanten_table[key]

                    shan=self.__shanten_ippan_sub(ji_nl,su_nl,naki_mentsu)-1
                    min_s=min(min_s,shan)
        
        return min_s


    def shanten_kokushi(self,hand27):
        toitsu=0
        sk=13
        idxs=[0,1,2,10,11,19,20,21,22,23,24,25,26]
        for idx in idxs:
            if hand27[idx]>0:
                sk-=1
                if hand27[idx]>1 and toitsu==0:
                    toitsu=1
        return sk-toitsu

    def shanten_chitoi(self,hand27,four_ok=False):
        toitsu_suu=0
        syurui_suu=0
        all_shurui_suu=0
        for i in hand27:
            if i>=1:
                all_shurui_suu=all_shurui_suu+1
                if i>=2:
                    toitsu_suu=toitsu_suu+i//2
                    syurui_suu=syurui_suu+1
        if four_ok:
            shanten_c=6-toitsu_suu
        else:
            all_shurui_suu=7 if all_shurui_suu>7 else all_shurui_suu
            shanten_c = 6-syurui_suu+ 7-all_shurui_suu
        return shanten_c

    ###役の計算,yaku=[0or1]*52,han,fuを返す:double_yakuman~は14~
            #(in yakuman_check())
            #if yakuman:cut yaku not in yakuman 
            #elif not yakuman:han=min(han,13)  

    
    def __yaku_han_push(self,yaku_str,yaku,han):
        yaku[self.yaku_dic_str[yaku_str][0]]=1
        han+=self.yaku_dic_str[yaku_str][1]
        return yaku,han
    def yakuman_check(self,yaku,han):
        #複合しない役満はyaku構築の実装上でも複合しない
        nums=[0]*2
        new_yaku=[0]*self.yaku_length
        flag=False
        for y in self.yakuman_set:
            yaku_id,h=self.yaku_dic_str[y]
            if yaku[yaku_id]:
                if h==13:nums[0]+=1
                else:nums[1]+=1
                new_yaku[yaku_id]=1
                flag=True
        if flag:
            return new_yaku,12+nums[0]+nums[1]*2
        else:
            return yaku,min(han,13)
    def sanshoku_doko_check(self,yaku,han,hand108):
        idxs=[[0,2,11],[1,10,19]]
        for i in range(2):
            num=0
            for idx in idxs[i]:
                for j in range(4):
                    if j==0:n=3
                    else:n=1
                    if hand108[idx+27*j]>=n:
                        num+=1
            if num==3:
                hand27=np.copy(hand108[:27])
                for idx in idxs[i]:
                    if hand27[idx]:
                        hand27[idx]-=3
                if self.shanten_ippan(hand27,naki_mentsu=(14-sum(hand27))//3)==-1:
                    yaku,han=self.__yaku_han_push('sanshoku_doko',yaku,han)
                    break
        return yaku,han
    def tanyao_check(self,yaku,han,hand108):
        for pai in self.terminal_idxs:
            for j in range(4):
                if hand108[pai+27*j]:
                    return yaku,han
        for pai in self.jihai_idx27:
            for j in range(4):
                if hand108[pai+27*j]:
                    return yaku,han
        yaku,han=self.__yaku_han_push('tanyao',yaku,han)
        return yaku,han
    def __roto_check(self,yaku,han,hand108):
        num19=0
        for pai in self.terminal_idxs:
            for j in range(4):
                if hand108[pai+27*j]:
                    if j==0:num19+=hand108[pai]
                    else:num19+=3
        if num19==14:
            yaku,han=self.__yaku_han_push('chinroto',yaku,han)
            return yaku,han
        
        num_ji=0
        for pai in self.jihai_idx27:
            for j in range(4):
                if hand108[pai+27*j]:
                    if j==0:num_ji+=hand108[pai]
                    else:num_ji+=3
        if num_ji+num19==14:
            yaku,han=self.__yaku_han_push('honroto',yaku,han)
        return yaku,han
    def __somete_check(self,yaku,han,hand108,menzen):
        #tsuiso,ryuiso,chinitsu,honitsu
        num_kind=0
        num=0
        ryuiso=True

        for kind in range(3):
            not_find=True
            length=len(self.kind_idxs[kind])
            for i in range(length):
                pai=self.kind_idxs[kind][i]
                for j in range(4):
                    if hand108[pai+27*j]:
                        if not_find:
                            num_kind+=1
                            not_find=False
                        if num_kind>1:
                            return yaku,han

                        if j==0:num+=hand108[pai]
                        else:num+=3
                        if pai in self.not_ryuiso:ryuiso=False
        if num==0:
            yaku,han=self.__yaku_han_push('tsuiso',yaku,han)
        else:
            if ryuiso:
                r=0
                for j in range(4):
                    hatsu=25
                    if hand108[hatsu+27*j]:
                        if j==0:r+=hand108[hatsu+27*j]
                        else:r+=3
                if num+r==14:
                    yaku,han=self.__yaku_han_push('ryuiso',yaku,han)
                    return yaku,han
            if num==14:
                if menzen:yaku,han=self.__yaku_han_push('chinitsu',yaku,han)
                else:yaku,han=self.__yaku_han_push('chinitsu_naki',yaku,han)
            else:
                if menzen:yaku,han=self.__yaku_han_push('honitsu',yaku,han)
                else:yaku,han=self.__yaku_han_push('honitsu_naki',yaku,han)
        return yaku,han
    def __jipai_yaku_check(self,yaku,han,hand108,player_wind,field_wind):
        #sho_daisushi,sho_daisangen,yakuhai
        #sho_daisushi
        nums=[0]*2
        idxs=[20,21,22,23]
        for idx in idxs:
            for j in range(4):
                if j==0:
                    if hand108[idx]==2:
                        nums[0]+=1
                    elif hand108[idx]>2:
                        nums[1]+=1
                else:
                    if hand108[idx+j*27]:nums[1]+=1
        if nums[1]>=3:
            if nums[1]==4:
                yaku,han=self.__yaku_han_push('daisushi',yaku,han)
                return yaku,han
            elif nums[0]:
                yaku,han=self.__yaku_han_push('shosushi',yaku,han)
                return yaku,han
        
        #sho_daisangen
        nums=[0]*2
        idxs=[24,25,26]
        for idx in idxs:
            for j in range(4):
                if j==0:
                    if hand108[idx]==2:
                        nums[0]+=1
                    elif hand108[idx]>2:
                        nums[1]+=1
                else:
                    if hand108[idx+j*27]:nums[1]+=1
        if nums[1]>=2:
            if nums[1]==3:
                yaku,han=self.__yaku_han_push('daisangen',yaku,han)
                return yaku,han
            elif nums[0]:
                #yakuhaiと複合
                yaku,han=self.__yaku_han_push('shosangen',yaku,han)

        #yakuhai
        yakuhai=[20+field_wind,20+player_wind,24,25,26]
        for i in range(5):
            pai=yakuhai[i]
            yaku_str=self.jihai_str[pai-20]
            if i==0:
                yaku_str+='_ba'
            elif i==1:
                yaku_str+='_ji'
            for j in range(4):
                if j==0:
                    num=3
                else:
                    num=1
                if hand108[pai+27*j]>=num:
                    yaku,han=self.__yaku_han_push(yaku_str,yaku,han)
        return yaku,han
    def find_dora(self,hand116,dora_hyoji29,uradora_hyoji29=[],hand108=[]):
        if len(hand108)==0:
            hand108=self.transform_hand116to108(hand116)
        #akadora,nukidoraは前提
        doras=[0]*4#[表ドラ,裏ドラ,赤ドラ,抜きドラ]

        for pai29 in dora_hyoji29:
            pai27=self.transform_pai29to27(pai29)
            next_pai=self.next_pai27(pai27)
            for j in range(4):
                if hand108[next_pai+27*j]:
                    if j==0:doras[0]+=hand108[next_pai]
                    else:doras[0]+=(j+6)//2
            if next_pai==23:#ドラが北のとき
                doras[0]+=hand116[self.hand116_nuki_idx]
                    
        for pai29 in uradora_hyoji29:
            pai27=self.transform_pai29to27(pai29)
            next_pai=self.next_pai27(pai27)
            for j in range(4):
                if hand108[next_pai+27*j]:
                    if j==0:doras[1]+=hand108[next_pai]
                    else:doras[1]+=(j+6)//2
            if next_pai==23:#ドラが北のとき
                doras[1]+=hand116[self.hand116_nuki_idx]
        
        for pai116 in self.hand116_akadora_idxs:
            doras[2]+=hand116[pai116]
        doras[3]+=hand116[self.hand116_nuki_idx]
        return doras



    #mj_module用に作り変えている
    def yaku_check_mjmodule(self,hand116,win_tile29,tsumo,player_wind,field_wind,dora_hyoji29,uradora_hyoji29,last_tsumo,rinshan,chankan,hand108=[]):
        if len(hand108)==0:
            hand108=self.transform_hand116to108(hand116)
        doras=self.find_dora(hand116,dora_hyoji29,uradora_hyoji29,hand108=hand108)
        win_tile27=self.transform_pai29to27(win_tile29)
        shanten_i,shanten_c,shanten_k=self.shanten(hand108)
        yaku=[0]*self.yaku_length
        han=0
        fu=0
        if min(shanten_i,shanten_c,shanten_k)==-1:
            if shanten_k==-1:
                if hand108[win_tile27]==2:
                    yaku,han=self.__yaku_han_push('kokushi13',yaku,han)
                else:
                    yaku,han=self.__yaku_han_push('kokushi',yaku,han)
                
                #tenho
                if tsumo and hand116[self.hand116_ippatsu_idx] and not hand116[self.hand116_reach_idx]:
                    yaku,han=self.__yaku_han_push('tenho',yaku,han)
            else:
                #tenho
                if tsumo and hand116[self.hand116_ippatsu_idx] and not hand116[self.hand116_reach_idx]:
                    yaku,han=self.__yaku_han_push('tenho',yaku,han)
                han+=sum(doras)
                if hand116[self.hand116_reach_idx]>0:
                    if hand116[self.hand116_reach_idx]==1: yaku,han=self.__yaku_han_push('reach',yaku,han)
                    elif hand116[self.hand116_reach_idx]==2: yaku,han=self.__yaku_han_push('double_reach',yaku,han)
                    if hand116[self.hand116_ippatsu_idx]:yaku,han=self.__yaku_han_push('ippatsu',yaku,han)
                if chankan: yaku,han=self.__yaku_han_push('chankan',yaku,han)
                if rinshan: yaku,han=self.__yaku_han_push('rinshan_kaiho',yaku,han)
                if last_tsumo==0 and not rinshan:
                    if tsumo: yaku,han=self.__yaku_han_push('haitei',yaku,han)
                    else: yaku,han=self.__yaku_han_push('hotei',yaku,han)
                if tsumo and hand116[self.hand116_menzen_idx]: yaku,han=self.__yaku_han_push('mentsumo',yaku,han)
                #chitoiと複合する役集合:somete_check
                    #緑一色(七対子と複合はしないけど)*,字一色,(天和:複合するけど別check)　　混老頭、混一色、清一色
                    #tanyao
                yaku,han=self.__somete_check(yaku,han,hand108,hand116[self.hand116_menzen_idx])
                yaku,han=self.__roto_check(yaku,han,hand108)
                if shanten_i==-1:
                    if self.shanten_chitoi(hand108[:27],four_ok=True)==-1:
                        yaku,han=self.__yaku_han_push('ryanpeko',yaku,han)
                    yaku,han=self.__jipai_yaku_check(yaku,han,hand108,player_wind,field_wind)
                    yaku,han=self.sanshoku_doko_check(yaku,han,hand108)
                    
                    hand_mjmodule=self.transform_hand108tomjmodule(hand108)
                    win_tile=self.transform_pai27to34(win_tile27)*4
                    self.config_mj_module.is_tsumo=tsumo
                    self.config_mj_module.player_wind=27+player_wind
                    self.config_mj_module.round_wind=27+field_wind
                    result_mjmodule=self.calculator_mjmodule.estimate_hand_value(hand_mjmodule[0],win_tile,melds=hand_mjmodule[1],config=self.config_mj_module)
                    yaku,han=self.__yaku_han_push_mjmodule(result_mjmodule,yaku,han,hand116[self.hand116_menzen_idx])
                    fu=result_mjmodule.fu
                elif shanten_c==-1:
                    yaku,han=self.__yaku_han_push('chitoitsu',yaku,han)
                    fu=25

                    #chitoiと複合する役集合
                    yaku,han=self.tanyao_check(yaku,han,hand108)
        yaku,han=self.yakuman_check(yaku,han)
        return (yaku,han,fu,doras)
    
    def transform_hand108tomjmodule(self,hand108):
        hand=[0]*34
        melds=[]
        for i in range(27):
            if hand108[i]:
                hand[self.transform_pai27to34(i)]=hand108[i]
        meld_types=[Meld.PON,Meld.KAN,Meld.KAN]
        opens=[True,True,False]
        for i in range(3):
            for j in range(27):
                if hand108[27*(i+1)+j]:
                    hand[self.transform_pai27to34(j)]+=3
                    melds.append(Meld(meld_type=meld_types[i], tiles=[self.transform_pai27to34(j)*4+k for k in range((i+7)//2)],opened=opens[i]))
        hand=TilesConverter.to_136_array(hand)
        return [hand,melds]
    
    def __yaku_han_push_mjmodule(self,result_mjmodule,yaku,han,menzen):
        if result_mjmodule.error:
            if result_mjmodule.error=='There are no yaku in the hand':
                pass
            #print(f'error in yaku_mjmodule')
        else:
            yaku_str_list=list(map(str,result_mjmodule.yaku))
            kuisagari_index={4,6,8}
            for y in yaku_str_list:
                if y in self.yaku_dic_mjmodule.keys():
                    i=self.yaku_dic_mjmodule[y]
                    if (not menzen) and i in kuisagari_index:i+=1
                    s=self.yaku_dic_int[i]
                    yaku,han=self.__yaku_han_push(s,yaku,han)
        return yaku,han
    def machi(self,hand108):
        l=[0]*27

        ch=np.copy(hand108[:27])
        naki_mentsu=np.sum(hand108[27:])
        for i in range(27):
            ch[i]+=1
            shanten=self.shanten(ch,naki_mentsu=naki_mentsu)
            if min(shanten)==-1:
                l[i]=1
            ch[i]-=1
        return l