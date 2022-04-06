import numpy as np
import my_yaku_module

class vectorise_controller:
    def __init__(self):

        self.mym=my_yaku_module.my_yaku_module()

        self.length=755
        self.info_idx=0
        self.info_length=44
        self.kawa_idx=44
        self.kawa_length=29
        self.hand_idx=131
        self.hand_length=116
        self.action_idx=479
        self.action_length=92

        self.info_dora_idx=15

    def transform(self,array,hidden_player):
        
        xl=[]
        yl=[]
        kawa_p=np.array([0]*29)
        for i in range(array.shape[0]):
            if array[i,12+hidden_player]:
                #action_check
                #handをツモる直前にする
                #ツモあがりはcut
                if array[i,self.action_idx+self.action_length*hidden_player+86]:
                    continue
                if array[i,self.action_idx+self.action_length*hidden_player+29]:
                    #nuki
                    array[i,self.hand_idx+self.hand_length*hidden_player+23]-=1
                #dahai
                for pai in range(29):
                    if array[i,self.action_idx+self.action_length*hidden_player+pai]:
                         array[i,self.hand_idx+self.hand_length*hidden_player+pai]-=1
                #ankan
                for pai in range(27):
                    if array[i,self.action_idx+self.action_length*hidden_player+30+pai]:
                        if pai==6 or pai==15:
                            red_pai=27+pai//10
                            array[i,self.hand_idx+self.hand_length*hidden_player+pai]-=3
                            array[i,self.hand_idx+self.hand_length*hidden_player+red_pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_ankan_sidx+pai]+=1
                        else:
                            array[i,self.hand_idx+self.hand_length*hidden_player+pai]-=4
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_ankan_sidx+pai]+=1
                #kakan
                for pai in range(29):
                    if array[i,self.action_idx+self.action_length*hidden_player+57+pai]:
                        if pai==27 or pai==28:
                            normal_pai=[6,15][pai-27]
                            array[i,self.hand_idx+self.hand_length*hidden_player+pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_pon_sidx+normal_pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_minkan_sidx+normal_pai]+=1
                        if pai==6 or pai==15:
                            red_pai=27+pai//10
                            array[i,self.hand_idx+self.hand_length*hidden_player+pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_pon_sidx+red_pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_minkan_sidx+pai]+=1
                        else:
                            array[i,self.hand_idx+self.hand_length*hidden_player+pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_pon_sidx+pai]-=1
                            array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_minkan_sidx+pai]+=1

                #ダブリーは消去
                for i in range(3):
                    if array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_reach_idx]==2:
                        array[i,self.hand_idx+self.hand_length*hidden_player+self.mym.hand116_reach_idx]=1

                kawa_n=np.array([0]*29)
                for i in range(3):
                    kawa_n+=array[i,self.kawa_idx+self.kawa_length*hidden_player:self.kawa_idx+self.kawa_length*(hidden_player+1)].astype(int)
                xl.append(self.transform_x(array[i],hidden_player,kawa_n-kawa_p))
                yl.append(self.transform_y(array[i],hidden_player))
                kawa_p=kawa_n
        

        return np.stack(xl),np.stack(yl)
                
    def transform_x(self,array,hidden_player,passed_tiles):
        xl=[]
        xl.append(self.last_pai(array,hidden_player))

        for i in range(3):
            if array[i]:
                field_wind=i
        for i in range(3):
            if array[3+i]:
                kyoku=i
        xl.append(self.public_hand(array[self.hand_idx+self.hand_length*hidden_player:self.hand_idx+self.hand_length*(hidden_player+1)].astype(int),field_wind,(hidden_player-kyoku)%3))

        for i in range(2):
            player_id=(hidden_player+1+i)%3
            xl.append(self.public_hand(array[self.hand_idx+self.hand_length*player_id:self.hand_idx+self.hand_length*(player_id+1)].astype(int),field_wind,(player_id-kyoku)%3))
        
        #kawa
        kawa=np.eye(5)[array[self.kawa_idx+self.kawa_length*hidden_player:self.kawa_idx+self.kawa_length*(hidden_player+1)].astype(int)]
        xl.append(kawa.flatten())

        #info
        info=np.zeros(40)
        idx=0#自風
        info[idx+field_wind]=1
        idx=3#場風
        info[idx+(hidden_player-kyoku)%3]=1
        idx=6#kyoku
        info[idx+kyoku]=1
        idx=9#持ち点
        for i in range(3):
            info[idx+i]=array[8+i]/3.5
        idx=12#残り山
        info[idx]=array[11]/100
        idx=13
        for i in range(29):
            if array[15+i]:
                pai=self.mym.transform_pai29to27(i)
                pai=self.mym.next_pai27(pai)
                info[idx+pai]+=array[15+i]
        xl.append(info)

        #action
        idxs=list(range(86))+[87,91]
        idxs=list(map(lambda x:self.action_idx+self.action_length*hidden_player+x,idxs))
        xl.append(array[idxs])

        #passed_tiles
        xl.append(passed_tiles)
        
        return np.concatenate(xl)
    def transform_y(self,array,hidden_player):
        yl=[]
        hand116=array[self.hand_idx+self.hand_length*hidden_player:self.hand_idx+self.hand_length*(hidden_player+1)].astype(int)
        hand108=self.mym.transform_hand116to108(hand116)

        for i in range(3):
            if array[i]:
                field_wind=i
        for i in range(3):
            if array[3+i]:
                kyoku=i
        player_wind=(hidden_player-kyoku)%3

        #牌種類
        pais=np.eye(5)[hand116[:29]]
        yl.append(pais.flatten())

        #shuntsu
        shuntsu=np.zeros(14)
        for i in range(2):
            sidx=[2,11][i]
            for j in range(7):
                if np.all(hand116[sidx+j:sidx+3+j]>1):
                    shuntsu[i*7+j]=1
        yl.append(shuntsu)

        #dora
        dora_hyoji=[]
        for i in range(29):
            dora_hyoji+=[i]*int(array[15+i])
        
        doras=self.mym.find_dora(hand116,dora_hyoji)
        dora=sum(doras)
        yl.append(np.eye(13)[dora])


        #yaku
        yl.append(self.yaku_main(hand116,field_wind,player_wind))

        #tenpai
        shanten=self.mym.shanten(hand108)
        machi=self.mym.machi(hand108)
        machi=np.array(machi)
        tenpai=np.array([np.any(machi)])

        if hand116[self.mym.hand116_menzen_idx]:
            if hand116[self.mym.hand116_reach_idx]:
                tsumoron_ratio=[0.44,0.56]
            else:
                tsumoron_ratio=[0.58,0.42]
        else:
            tsumoron_ratio=[0.55,0.45]


        han_tsumo=np.zeros(13)
        han_ron=np.zeros(13)

        for i in range(27):
            if machi[i]:
                _,han,_,_=self.mym.yaku_check_mjmodule(hand116,i,1,player_wind,field_wind,[],[],5,0,0,hand108=hand108)
                han=min(max(0,han-1),12)#-1して0~12
                han_tsumo[han]+=tsumoron_ratio[0]
                
                _,han,_,_=self.mym.yaku_check_mjmodule(hand116,i,0,player_wind,field_wind,[],[],5,0,0,hand108=hand108)
                han=min(max(0,han-1),12)
                han_ron[han]+=tsumoron_ratio[1]
        
        hans=han_tsumo+han_ron
        if np.sum(hans)>0:
            hans/=np.sum(hans)

        yl.append(machi)
        yl.append(hans)
        yl.append(tenpai)
        return np.concatenate(yl)
    def last_pai(self,array,hidden_player):
        """
        return [145]=[29,5]_onehot.flatten()
        """
        last=np.array([4]*29)
        last[27]=1
        last[28]=1
        #dora
        last-=array[self.info_dora_idx:self.info_dora_idx+29].astype(int)
        #kawa
        for i in range(3):
            last-=array[self.kawa_idx+self.kawa_length*i:self.kawa_idx+self.kawa_length*(i+1)].astype(int)
        #hand
        for i in range(3):
            #非公開
            if not i==hidden_player:
                last-=array[self.hand_idx+self.hand_length*i:self.hand_idx+self.hand_length*i+29].astype(int)
            #公開
            
            #北
            last[23]-=array[self.hand_idx+self.hand_length*i+self.mym.hand116_nuki_idx].astype(int)
            #pon
            for pai in range(29):
                if array[self.hand_idx+self.hand_length*i+self.mym.hand116_pon_sidx+pai]:
                    if pai==27 or pai==28:
                        normal_pai=[6,15][pai-27]
                        last[normal_pai]-=2
                        last[pai]-=1
                    else:
                        last[pai]-=3
            #minkan,ankan
            for j in range(2):
                for pai in range(27):
                    if array[self.hand_idx+self.hand_length*i+self.mym.hand116_minkan_sidx+27*j+pai]:
                        if pai==6:
                            last[pai]-=3
                            last[27]-=1
                        elif  pai==15:
                            last[pai]-=3
                            last[28]-=1
                        else:
                            last[pai]-=4
        last=np.eye(5)[last]
        return last.flatten()
    def public_hand(self,hand116,field_wind,player_wind):
        l=[]
        idxs=list(range(self.mym.hand116_pon_sidx,self.mym.hand116_ippatsu_idx))
        l.append(hand116[idxs])

        #nuki
        nuki=np.eye(5)[hand116[self.mym.hand116_nuki_idx]]
        l.append(nuki)

        #役牌
        a=[0]*5
        for k in range(5):
            pai=[20+field_wind,20+player_wind,24,25,26][k]
            for sidx in [self.mym.hand116_pon_sidx,self.mym.hand116_minkan_sidx,self.mym.hand116_ankan_sidx]:
                if hand116[sidx+pai]:
                    a[k]=1
                    continue
        a=np.array(a)
        l.append(a)

        return np.concatenate(l)
    def yaku_main(self,hand116,field_wind,player_wind):
        hand108=self.mym.transform_hand116to108(hand116)
        yaku_length=10
        yaku=[0]*yaku_length

        menzen=hand116[self.mym.hand116_menzen_idx]

        yakuhai=self.yakuhai_check(hand108,field_wind,player_wind)
        yaku[0:2]=yakuhai
        yaku[2]=self.tanyao_check(hand108)
        yaku[3]=self.toitoi_check(hand108)
        yaku[4]=self.chitoi_check(hand108,menzen)
        yaku[5:9]=self.somete_check(hand108)
        yaku[9]=self.kokushi_check(hand108,menzen)
        return np.array(yaku)


    def yakuhai_check(self,hand108,field_wind,player_wind):
        ans=[0]*2
        for pai in [20+field_wind,20+player_wind,24,25,26]:
            for j in range(4):
                if j==0:
                    if hand108[pai]>1:
                        if hand108[pai]==2:
                            ans[0]=1
                        else:
                            ans[1]=1
                        break
                else:
                    ans[1]=1
                    break
        return ans
    
    def tanyao_check(self,hand108):
        for pai in self.mym.terminal_idxs:
            for j in range(4):
                if hand108[pai+27*j]:
                    return 0
        for pai in self.mym.jihai_idx27:
            for j in range(4):
                if hand108[pai+27*j]:
                    return 0
        return 1
    
    def toitoi_check(self,hand108):
        minko=sum(hand108[27:])
        anko=np.sum(hand108[:27]>2)
        toitsu=np.sum(hand108[:27]>1)
        if anko and toitsu+minko>4:
            return 1
        else:
            return 0
    def chitoi_check(self,hand108,menzen):
        if not menzen:
            return 0
        else:
            return int(np.sum(hand108>1)>4)
    
    def somete_check(self,hand108):
        kind=[0]*4
        for i in range(4):
            kind_idxs=self.mym.kind_idxs[i]
            for j in range(4):
                if np.sum(hand108[27*i:27*(i+1),kind_idxs])>0:
                    kind[i]=1
        ans=[0]*4
        if kind[0] or (kind[1] and kind[2]):
            return ans
        elif kind[1]:
            if kind[3]:
                ans[0]=1
            else:
                ans[1]=1
        elif kind[2]:
            if kind[3]:
                ans[2]=1
            else:
                ans[3]=1
        else:
            return ans
    def kokushi_check(self,hand108,menzen):
        if not menzen:
            return 0
        else:
            a=0
            a+=np.sum(hand108[self.mym.terminal_idxs])
            a+=np.sum(hand108[self.mym.jihai_idx27])
            if a>8:
                return 1
            else:return 0




            




"""
環境ベクトルの長さはxの1次元ベクトル

情報ベクトル:44次元
0~43
河ベクトル:87(29*3人)次元
44~72
73~101
102~130
手牌ベクトル:348(116*3人)次元
131~246
247~361
362~478
	ここまで全体で479
+
行動ベクトル276(92*3)次元
479~754

全部で
755次元

1局ごとに終局ベクトル


環境全体
	ツモ番毎に記録
	全体として綺麗なベクトル(行列表現)にはならないけどとりあえず問題なし

情報ベクトル:
	44次元
	one-hot化する.

	場風:3次元(onehot)
	自風:0,とりあえず含まない、学習時にそれぞれのagentごとにつける?
	局:3次元(one_hot)
	本場:1次元
	供託:1次元
	持ち店/10000:3次元#起家の表現もいれる
	残りツモ回数:1次元(直後のツモを含む,最後の切り番で0)
	誰のキリ番か:3次元(onehot)
	ドラ表示:29次元(順番無視)
		学習時に枚数でonehotにする?
		or
		順番で29*4にする?

	0.1,2:場風
	3.4.5:局
	6:本場
	7:供託
	8.9.10:持ち点
	11:残りツモ
	12,13,14:切り番
	15~43:ドラ
河ベクトル
	87次元(29次元*3人)
	
	順番考慮せずに種類と枚数で表現する。
	problem
		(ツモぎり、手出し,鳴かれたかどうか)の3チャネル
			順番を考慮しないなら河では表現できない
				→現状しない
				→差異とactionベクトルで表現できてると信じたい
手牌ベクトル
	348=116*3次元
	手牌29+抜き1+pon29+明カン27+暗カン27+面前1+リーチ1=115次元
	公開部分:手牌29次元以外
	誰から鳴いたかは表現しない(いら無さそうだし情報としては河に残したい(サンマはかぶらないので残す必要がない。))
	0~28:手牌
	29:抜き
	30~58:pon
	59~85:加カン,大明カン,同様に表現
	86~112:暗カン
	113:面前
	114:リーチ:(ダマ:0,ノーマル:1,ダブリー:2)
	115:一発の権利:
		初期1,リーチすると1になる
		打牌,他の人の鳴きで0になる
		(reach and ipp)でipatsu,(tsumo and ipp and not reach)で天地
		
	
	学習時に保留
		hand部分をmask,0,1,2,3,4の6次元onehot
			構造的になんとなく赤も形を合わせてみたい
		抜き部分0,1,2,3,4の5次元onehot
		自風をonehotで追加


行動ベクトル
	276次元
	(切り29,抜き1,アンカン27,カカン29,ツモ上り1,ツモ切り1,ポンされる1,大明カンされる1,ロンされる1,リーチする1)*3=92*3=276
	0~28:切り
	29:抜き
	30~56:アンカン
	57~85:カカン
	86:ツモ上り
	87:ツモ切り
	88:ポンされる
	89:大明カンされる
	90:ロンされる
	91:リーチする


	学習時に一つ前のを渡す？
	鳴かれた時の学習は切るまでは普段どうりで2段階にする？


終局情報
	取り扱いが難しいからとりあえずそのままjson形式で渡してみる



ベクトル抽出
actionベクトルは一つ前のを更新

tsumo
	hand更新
	ツモ回数更新
	切り番更新
dahai
	hand抽出
	hand更新
	kawa更新
	打牌のaction更新
pon
	ponされたaction更新
	hand更新
	切り番更新
nuki
	hand抽出
	hand更新
	action更新
kakan
	hand抽出
	hand更新
	ドラの更新はtype:dora
	action更新
ankan
	hand抽出
	hand更新
	ドラの更新はtype:dora
	action更新
daiminkan
	カンされたaction更新
	hand更新
	ドラの更新はtype:dora
reach1
	step2がない場合もあるから必要
	リーチonのhand更新情報をdahaiへpass
	action更新情報をdahaiへpass
reach2
	供託、持ち点更新
agari
	ツモ
		hand push
		actionの更新
	ロン
		actionの更新
dora
	ドラの更新
	カンの後に別で処理されてる
	明カンと暗カンでタイミングが別

仕様
	９種はcut

"""

