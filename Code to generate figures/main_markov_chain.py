import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 

def algo_forward(L_etat, L_seq_test, Pb_init, Pb_final, \
                 A_trans, B_emis, dico_seq):
    M_val_fwd = np.zeros((len(L_etat), len(L_seq_test)),dtype=np.float64)

    for i, sequence_val in enumerate(L_seq_test):
        for j in range(len(L_etat)):
            if (i == 0):
                M_val_fwd[j, i] = Pb_init[j] * B_emis[j, dico_seq[sequence_val]]
            else:
                val = [M_val_fwd[k, i - 1] * B_emis[j, dico_seq[sequence_val]] * A_trans[k, j] for k in
                          range(len(L_etat))]
                M_val_fwd[j, i] = sum(val)

    #end state value
    val_fwd = sum(np.multiply(M_val_fwd[:, -1], Pb_final))
    return M_val_fwd, val_fwd

def algo_backward(L_etat:list, L_seq_test:list, Pb_init:list, Pb_final:list,
                  A_trans:'np.array', B_emis:'np.array', dico_seq:dict):
    M_val_bwd = np.zeros((len(L_etat), len(L_seq_test)),dtype=np.longdouble)

    for i in range(1,len(L_seq_test)+1):
        for j in range(len(L_etat)):
            if (i == 1):
                M_val_bwd[j, -i] = Pb_final[j]
            else:
                val = [M_val_bwd[k, -i+1] * \
                       B_emis[k, dico_seq[L_seq_test[-i+1]]] * \
                       A_trans[j, k] for k in range(len(L_etat))]
                M_val_bwd[j, -i] = sum(val)

    #start state value
    val = [M_val_bwd[m,0] *
           B_emis[m, dico_seq[L_seq_test[0]]] \
           for m in range(len(L_etat))]
    val_bwd = sum(np.multiply(val, Pb_init))
    return M_val_bwd, val_bwd

def Pb_si(M_val_fwd:'np.array', M_val_bwd:'np.array', val_fwd:float,
          L_etat:list, L_seq_test:list,
          A_trans:'np.array', B_emis:'np.array', dico_seq:dict):
    M_pb_si = np.zeros((len(L_etat), len(L_seq_test)-1, \
                        len(L_etat)),dtype=np.longdouble)

    for i in range(len(L_seq_test)-1):
        for j in range(len(L_etat)):
            for k in range(len(L_etat)):
                M_pb_si[j, i, k]= (M_val_fwd[j,i]*M_val_bwd[k,i+1]*\
                                   A_trans[j,k]*\
                                   B_emis[k, dico_seq[L_seq_test[i+1]]])\
                                   /(val_fwd+1.e-6)
    return M_pb_si

def Pb_gamma(M_val_fwd:'np.array', M_val_bwd:'np.array', val_fwd:float, \
             L_etat:list, L_seq_test:list):
    M_pb_gamma = np.zeros((len(L_etat), len(L_seq_test)),\
                          dtype=np.longdouble)

    for i in range(len(L_seq_test)):
        for j in range(len(L_etat)):
            M_pb_gamma[j, i] = (M_val_fwd[j, i] * M_val_bwd[j, i])\
                               / (val_fwd+1e-15)

    return M_pb_gamma

class traitement_des_donnees:
    # Conversion pixel mm : 100 pixels -> 22 mm
    def __init__(self, lien:str)->bool:
            with open(lien, 'r') as f:
                ind = 0
                for l in f:
                    if ind == 0:
                        t_ouverture = eval(l)
                        ind += 1
                        if type(t_ouverture) == list:
                            L = t_ouverture 
                            t_ouverture = 0
                            ind += 1
                    elif ind == 1:
                        L = eval(l)
                        ind += 1
                    else:
                        self._L_t_max = eval(l)
            L_temps = [elmt[1] for elmt in L]

            L_detecteur = [elmt[2] for elmt in L]
            L_profil = [elmt[3] for elmt in L]

            self.func_donnee_filtrage(L_temps, L_detecteur, L_profil, t_ouverture)
            
    def func_donnee_filtrage(self, L_temps:list, L_detecteur:list, L_profil:list,
                             t_ouverture:float=0):
        L_temps, L_detecteur, L_profil = self.func_filtrage(L_temps, L_detecteur,
                                                            L_profil, t_ouverture)

        L_temps_paquet = []
        L_temps_min_paquet = []
        L_temps_max_paquet = []
        L_detecteur_paquet = []
        self._L_taille_max_passage = []
        ind = 0
        self._L_ind = [0]
        i = 1
        while i < len(L_temps):
            if L_temps[i]-L_temps[i-1] < 0.06 and \
                L_detecteur[i] == L_detecteur[i-1] :
                i += 1
            else:
                if np.std([len(elmt) for elmt in L_profil[ind:i]]) < 0.7:
                    for j in range(i-1, ind-1, -1):
                        del L_temps[j]
                        del L_detecteur[j]
                        del L_profil[j]

                    i = ind+1
                
                else : 
                    L_temps_paquet.append(np.mean(L_temps[ind:i]))
                    L_temps_min_paquet.append(L_temps[ind])
                    L_temps_max_paquet.append(L_temps[i-1])
                    L_detecteur_paquet.append(np.mean(L_detecteur[ind:i]))
                    self._L_taille_max_passage.append(\
                        (22/100)*np.max([len(elmt) for elmt in L_profil[ind:i]]))
                    self._L_ind.append(i)

                    ind = i
                    i += 1

        L_temps_paquet.append(np.mean(L_temps[ind:len(L_temps)]))
        L_temps_min_paquet.append(L_temps[ind])
        L_temps_max_paquet.append(L_temps[len(L_temps)-1])
        L_detecteur_paquet.append(np.mean(L_detecteur[ind:len(L_temps)]))
        self._L_taille_max_passage.append(\
            (22/100)*np.max([len(elmt) for elmt in L_profil[ind:len(L_temps)]]))
            
        self._L_choix = [0]
        self._L_temps_paquet = [0]+L_temps_paquet
        self._L_temps_min_paquet = [0]+L_temps_min_paquet
        self._L_temps_max_paquet = [0]+L_temps_max_paquet

        self._L_profil = L_profil

        for i in range(len(L_detecteur_paquet)):
            if self._L_choix[len(self._L_choix)-1] == 0:
                self._L_choix.append(1 if L_detecteur_paquet[i]==2 else -1)
            else :
                if L_detecteur_paquet[i] != L_detecteur_paquet[i-1]:
                    self._L_choix[len(self._L_choix)-2] = 1 if L_detecteur_paquet[len(self._L_choix)-2]==2 else -1
                    self._L_choix[len(self._L_choix)-1] = 0
                    self._L_choix.append(1 if L_detecteur_paquet[i]==2 else -1)   
                else:
                    self._L_choix.append(0)

        self._L_t_aff_choix=[self._L_temps_paquet[0]]
        self._L_aff_choix=[self._L_choix[0]]
        for i in range(1,len(self._L_choix)):
            self._L_t_aff_choix.append(self._L_temps_min_paquet[i])
            self._L_t_aff_choix.append(self._L_temps_max_paquet[i])
            self._L_aff_choix.append(self._L_choix[i-1])
            self._L_aff_choix.append(self._L_choix[i])
        self._L_t_aff_choix.append(self._L_t_max[len(self._L_t_max)-1])
        self._L_aff_choix.append(self._L_choix[len(self._L_choix)-1])

        self._L_t_1 = [0]
        self._L_t_2 = [0]
        self._L_1_larg = [0]
        self._L_1_pos = [0]
        self._L_2_larg = [0]
        self._L_2_pos = [0]
        for i in range(len(L_temps)):
            if L_detecteur[i] == 1:
                self._L_t_1.append(L_temps[i])
                self._L_1_larg.append(len(L_profil[i])*22/100)
                self._L_1_pos.append(L_profil[i][0]*22/100)
            else:
                self._L_t_2.append(L_temps[i])
                self._L_2_larg.append(len(L_profil[i])*22/100)
                self._L_2_pos.append(L_profil[i][0]*22/100)

    def __del__(self):
        pass
    
    def func_filtrage(self, L_temps:list, L_detecteur:list,
                      L_profil:list, t0:int=0)->tuple: 
        i = 0
        while i < len(L_temps):
            cond = False

            if len(L_profil[i])>1:
                j = 1
                while j < len(L_profil[i]):
                    if L_profil[i][j] != L_profil[i][j-1]+1:
                        cond = True 
                        j = len(L_profil[i])
                    else :
                        j += 1

            if i>0 and i < len(L_temps)-1:
                if not(L_temps[i] - L_temps[i-1] < 0.06 and \
                    L_detecteur[i] == L_detecteur[i-1] and \
                    abs(L_profil[i][0] - L_profil[i-1][0]) < 5 and \
                    abs(len(L_profil[i]) - len(L_profil[i-1])) < 5) and \
                    not(L_temps[i+1] - L_temps[i] < 0.06 and \
                    L_detecteur[i] == L_detecteur[i+1] and \
                    abs(L_profil[i][0] - L_profil[i+1][0]) < 5 and \
                    abs(len(L_profil[i]) - len(L_profil[i+1])) < 5):
                    cond = True 
            elif i == len(L_temps)-1:
                if not(L_temps[i] - L_temps[i-1] < 0.06 and \
                    L_detecteur[i] == L_detecteur[i-1] and \
                    abs(L_profil[i][0] - L_profil[i-1][0]) < 5 and \
                    abs(len(L_profil[i]) - len(L_profil[i-1])) < 5):
                    cond = True
            elif i==0:
                if not(L_temps[i+1] - L_temps[i] < 0.06 and \
                    L_detecteur[i] == L_detecteur[i+1] and \
                    abs(L_profil[i][0] - L_profil[i+1][0]) < 5 and \
                    abs(len(L_profil[i]) - len(L_profil[i+1])) < 5):
                    cond = True

            if L_temps[i] < t0 or L_temps[i] > self._L_t_max[1]:
                cond = True
            
            if len(L_profil[i]) > 45:
                cond = True

            if cond:
                del L_temps[i]
                del L_detecteur[i]
                del L_profil[i]
                if i > 0:
                    i -= 1
            else:
                i+=1

        for i in range(len(L_temps)):
            L_temps[i] -= t0
        self._L_t_max[1] -= t0
        return L_temps, L_detecteur, L_profil

    def func_figure_trajectoire(self)->None:
        L_t_aff_choix=[self._L_temps_paquet[0]]
        L_aff_choix=[self._L_choix[0]]
        L_taille_passage_bestiole = [0]
        L_temps_du_passage = [0]
        L_t_vit = [0]
        for i in range(1,len(self._L_choix)):
            L_t_aff_choix.append(self._L_temps_min_paquet[i])
            L_t_aff_choix.append(self._L_temps_max_paquet[i])
            L_aff_choix.append(self._L_choix[i-1])
            L_aff_choix.append(self._L_choix[i])

            L_t_vit.append(self._L_temps_min_paquet[i])
            L_t_vit.append((self._L_temps_min_paquet[i]+\
                            self._L_temps_max_paquet[i])/2)
            L_t_vit.append(self._L_temps_max_paquet[i])
            L_temps_du_passage.append(0)
            L_temps_du_passage.append(self._L_temps_max_paquet[i]-\
                                      self._L_temps_min_paquet[i])
            L_temps_du_passage.append(0)


            L_taille_passage_bestiole.append(0)
            L_taille_passage_bestiole.append(self._L_taille_max_passage[i//3])
            L_taille_passage_bestiole.append(0)

        L_t_aff_choix.append(self._L_t_max[1])
        L_aff_choix.append(self._L_choix[len(self._L_choix)-1])
        L_t_vit.append(self._L_t_max[1])
        L_temps_du_passage.append(0)
        L_taille_passage_bestiole.append(0)
        
        plt.figure()
        plt.subplot(313)
        plt.plot(L_t_aff_choix, L_aff_choix)
        plt.ylabel('Bras')
        plt.xlabel('Temps (s)')
        if len(self._L_t_max) == 0:
            plt.xlim(0,max(np.max(L_t_aff_choix), np.max(L_t_vit)))
        else:
            plt.xlim(0,self._L_t_max[1])
        plt.grid(axis='x',zorder=0)
        plt.subplots_adjust(hspace=0.02)
        plt.subplot(312)
        plt.plot(L_t_vit, 1000*np.array(L_temps_du_passage))
        plt.ylabel('Temps de \npassage (ms)')
        if len(self._L_t_max) == 0:
            plt.xlim(0,max(np.max(L_t_aff_choix), np.max(L_t_vit)))
        else:
            plt.xlim(0,self._L_t_max[1])
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.grid(axis='x',zorder=0)
        plt.subplots_adjust(hspace=0.02)
        plt.subplot(311)
        plt.plot(L_t_vit, L_taille_passage_bestiole)    
        plt.ylabel('Largeur\n maximale (mm)')
        if len(self._L_t_max) == 0:
            plt.xlim(0,max(np.max(L_t_aff_choix), np.max(L_t_vit)))
        else:
            plt.xlim(0,self._L_t_max[1])
        plt.grid(axis='x',zorder=0)
        plt.subplots_adjust(hspace=0.02)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.show()

    def func_figure_lateralisation(self)->None:
        L_pixel_0_1 = []
        L_pixel_1_0 = []
        L_pixel_0__1 = []
        L_pixel__1_0 = []

        compt_img = -1
        choix = -1
        for i in range(len(self._L_profil)):
            for k in self._L_profil[i]:
                for j in range(1,len(self._L_ind)):
                    if i>=self._L_ind[j-1] and i<self._L_ind[j]:
                        if self._L_choix[j] == -1 and \
                           self._L_choix[j-1] == 0:
                            if choix != 0:
                                choix = 0
                                compt_img += 1
                            L_pixel_0__1.append(k)      
                        elif self._L_choix[j] == 0 and \
                             self._L_choix[j-1] == -1:
                            if choix != 1:
                                choix = 1
                                compt_img += 1
                            L_pixel__1_0.append(k)                   
                        elif self._L_choix[j] == 1 and \
                             self._L_choix[j-1] == 0:
                            if choix != 2:
                                choix = 2
                                compt_img += 1
                            L_pixel_0_1.append(k)
                        elif self._L_choix[j] == 0 and \
                             self._L_choix[j-1] == 1:
                            if choix != 3:
                                choix = 3
                                compt_img += 1
                            L_pixel_1_0.append(k)   
                        else:
                            print('bizarre')
                        break

        L_axe = [i for i in range(0,101)]
        plt.figure()
        plt.grid(axis='x', zorder=0)
        plt.hist(L_pixel_0_1 +L_pixel_1_0 +L_pixel_0__1 +L_pixel__1_0, color=(59/255,82/255,139/255), ec='k', bins = L_axe, label=r'Bras 0 $\rightarrow$ Bras -1')
        plt.hist(L_pixel_0_1 +L_pixel_1_0 +L_pixel__1_0, color=(33/255,145/255,140/255), ec='k', bins = L_axe, label=r'Bras -1 $\rightarrow$ Bras 0')
        plt.hist(L_pixel_0_1 +L_pixel_1_0, color=(92/255,200/255,99/255), ec='k', bins = L_axe, label=r'Bras 0 $\rightarrow$ Bras 1')
        plt.hist(L_pixel_1_0, color=(253/255,231/255,37/255), ec='k', bins = L_axe, label=r'Bras 1 $\rightarrow$ Bras 0')    
        plt.ylabel('Occurence')
        plt.xlabel('Pixel')
        plt.legend()
        plt.xlim(0,99)
        plt.gca().twiny()
        plt.xticks(np.linspace(0,22,6))
        plt.xlabel('Taille (mm)')
        plt.show()

    def func_figure_largeur_passage(self)->None:
        L_bras__1_0 =  [self._L_taille_max_passage[i] \
                        for i in range(len(self._L_taille_max_passage)) \
                        if self._L_choix[i+1] == 0 and self._L_choix[i] == -1]
        L_bras_0__1 =  [self._L_taille_max_passage[i] \
                        for i in range(len(self._L_taille_max_passage)) \
                        if self._L_choix[i+1] == -1 and self._L_choix[i] == 0]
        L_bras_0_1 =  [self._L_taille_max_passage[i] \
                       for i in range(len(self._L_taille_max_passage)) \
                       if self._L_choix[i+1] == 1 and self._L_choix[i] == 0]
        L_bras_1_0 =  [self._L_taille_max_passage[i] \
                       for i in range(len(self._L_taille_max_passage)) \
                       if self._L_choix[i+1] == 0 and self._L_choix[i] == 1]

           
        L_axe = [i*22/100 for i in range(0,int(110*np.max(L_bras__1_0 +L_bras_0__1 +L_bras_0_1 +L_bras_1_0)/22))]
        plt.figure()
        plt.grid(axis='x', zorder=0)
        plt.hist(L_bras__1_0 +L_bras_0__1 +L_bras_0_1 +L_bras_1_0, color=(59/255,82/255,139/255), ec='k', bins = L_axe, label=r'Bras 0 $\rightarrow$ Bras -1')
        plt.hist(L_bras_0_1 +L_bras_1_0 +L_bras__1_0, color=(33/255,145/255,140/255), ec='k', bins = L_axe, label=r'Bras -1 $\rightarrow$ Bras 0')
        plt.hist(L_bras_0_1 +L_bras_1_0, color=(92/255,200/255,99/255), ec='k', bins = L_axe, label=r'Bras 0 $\rightarrow$ Bras 1')
        plt.hist(L_bras_1_0, color=(253/255,231/255,37/255), ec='k', bins = L_axe, label=r'Bras 1 $\rightarrow$ Bras 0')    
        plt.ylabel('Occurence')
        plt.xlabel('Largeur maximale lors du passage (mm)') 
        plt.xlim(0, np.max(L_axe))
        plt.legend()
        plt.show()

    def func_proba_temps(self)->None:
        L_temps_bras__1 = []
        L_temps_bras_0 = []
        L_temps_bras_1 = []
        for i in range(1,len(self._L_choix)):
            if self._L_choix[i-1] == -1:
                L_temps_bras__1.append(self._L_temps_paquet[i]\
                                       -self._L_temps_paquet[i-1])
            elif self._L_choix[i-1] == 0:
                L_temps_bras_0.append(self._L_temps_paquet[i]\
                                      -self._L_temps_paquet[i-1])
            else:
                L_temps_bras_1.append(self._L_temps_paquet[i]\
                                      -self._L_temps_paquet[i-1])

        if len(self._L_t_max)>0:
            ind = len(self._L_choix)-1
            if self._L_choix[ind] == -1:
                L_temps_bras__1.append(self._L_t_max[1]\
                                       -self._L_temps_paquet[ind])
            elif self._L_choix[ind] == 0:
                L_temps_bras_0.append(self._L_t_max[1]\
                                      -self._L_temps_paquet[ind])
            else:
                L_temps_bras_1.append(self._L_t_max[1]\
                                     -self._L_temps_paquet[ind])
        if sum(L_temps_bras__1)+sum(L_temps_bras_0)+sum(L_temps_bras_1)>0:
           
            fac = 1.20*max([np.max([0]+L_temps_bras__1+L_temps_bras_0+L_temps_bras_1)])

            bein_bin = [i*fac/10 for i in range(10)]
            plt.figure()
            if len(L_temps_bras_0)+len(L_temps_bras_1)+len(L_temps_bras__1)>0:
                plt.hist(L_temps_bras_0+L_temps_bras_1+L_temps_bras__1,
                         bins = bein_bin,
                         color=(34/255,177/255,76/255),
                         ec="k", label='Bras -1')
            if len(L_temps_bras_0)+len(L_temps_bras_1)>0 :
                plt.hist(L_temps_bras_0+L_temps_bras_1,
                         bins = bein_bin,
                         color=(0/255,162/255,232/255),
                         ec="k", label='Bras 0')
            if len(L_temps_bras_1)>0:
                plt.hist(L_temps_bras_1,
                         bins = bein_bin,
                         color=(63/255,72/255,204/255),
                         ec="k", label='Bras 1')
            plt.xlabel('Durée dans un bras (s)')
            plt.ylabel('Occurence')
            plt.xlim(0,1.5*max([np.max([0]+L_temps_bras__1 + L_temps_bras_0 + L_temps_bras_1)]))
            plt.legend()
            plt.show()

    def func_chaine_de_markov(self)->None:
        L_t_aff_choix=[self._L_temps_paquet[0]]
        
        L_aff_choix=[self._L_choix[0]]
        for i in range(1,len(self._L_choix)):
            L_t_aff_choix.append(self._L_temps_min_paquet[i])
            L_t_aff_choix.append(self._L_temps_max_paquet[i])
            L_aff_choix.append(self._L_choix[i-1])
            L_aff_choix.append(self._L_choix[i])

        L_t_aff_choix.append(self._L_t_max[1])
        L_aff_choix.append(self._L_choix[len(self._L_choix)-1])
        
        nb = 2
        for i in range(2, len(L_aff_choix)):
            if L_aff_choix[i-nb] == L_aff_choix[i]:
                for j in range(1,nb):
                    L_aff_choix[i-j] = L_aff_choix[i]


        if max(L_t_aff_choix)>2000:
            L_seq_test = [i for i in range(500)]
            L_seq_test[0] = 1
            for i in range(1,len(L_seq_test)):
                pas_i = int(max(L_t_aff_choix)/1000)
                for j in range(1, len(L_t_aff_choix)):
                    if L_t_aff_choix[j] > i*pas_i > L_t_aff_choix[j-1]:
                        L_seq_test[i] = L_aff_choix[j]+1
                        break
        else :
            L_seq_test = [i for i in range(int(np.max(L_t_aff_choix)))]
            L_seq_test[0] = 1
            for i in range(1,len(L_seq_test)):
                for j in range(1, len(L_t_aff_choix)):
                    if L_t_aff_choix[j] > i > L_t_aff_choix[j-1]:
                        L_seq_test[i] = L_aff_choix[j]+1
                        break

        A_trans = np.array([[0.8,0.2,0],
                               [0.2,0.6,0.2],
                               [0,0.2,0.8]])
        B_emis = np.array([[1,0.,0.],
                             [0.,1,0.],
                             [0.,0.,1]])

        L_etat = ['bras_g','centre', 'bras_droit']
        L_seq = [0, 1, 2]

        dico_seq = {0:0,1:1,2:2}
        Pb_init = [0,1,0]
        Pb_final = [1/3,1/3,1/3]

        A_trans = np.array([[0.8,0.2,0],
                        [0.2,0.6,0.2],
                        [0,0.2,0.8]])
        
        for it in range(100):
            M_val_fwd, val_fwd = algo_forward(L_etat, L_seq_test, Pb_init, Pb_final, \
                         A_trans, B_emis, dico_seq)
            M_val_bwd, val_bwd = algo_backward(L_etat, L_seq_test, Pb_init, Pb_final, \
                          A_trans, B_emis, dico_seq)
            M_pb_si = Pb_si(M_val_fwd, M_val_bwd, val_fwd, L_etat, L_seq_test, \
                 A_trans, B_emis, dico_seq)
            M_pb_gamma = Pb_gamma(M_val_fwd, M_val_bwd, val_fwd, L_etat, L_seq_test)
            # Calcul des matrices a et b
            a = np.zeros((len(L_etat), len(L_etat)))
            b = np.zeros((len(L_etat), len(L_seq)))
            
            #'a' matrix
            for j in range(len(L_etat)):
                for i in range(len(L_etat)):
                    for t in range(len(L_seq_test)-1):
                        a[j,i] = a[j,i] + M_pb_si[j,t,i]

                    denom_a = [M_pb_si[j, t_x, i_x] \
                               for t_x in range(len(L_seq_test) - 1) \
                               for i_x in range(len(L_etat))]
                    denom_a = sum(denom_a)

                    if (denom_a == 0):
                        a[j,i] = 0
                    else:
                        a[j,i] /= denom_a
            #'b' matrix\n",
            for j in range(len(L_etat)): #L_etat
                for i in range(len(L_seq)): #seq
                    indices = [idx for idx, val in \
                               enumerate(L_seq_test) if val == L_seq[i]]
                    b[j, i] = sum( M_pb_gamma[j,indices] )
                    denom_b = sum( M_pb_gamma[j,:] )

                    if (denom_b == 0):
                        b[j,i] = 0
                    else:
                        b[j, i] /= denom_b

            A_trans = a
            B_emis = b
            
            M_nouv_fwd, val_nouv_fwd = algo_forward(L_etat, L_seq_test, Pb_init, Pb_final, 
                         A_trans, B_emis, dico_seq)
            
            diff = np.abs(val_fwd - val_nouv_fwd)/max(val_fwd, val_nouv_fwd)
            
            if diff < 1e-4:
                break
        
        if np.sum(A_trans) == 0:
            A_trans = np.zeros((3,3))
            for i in range(1,len(L_seq_test)):
                A_trans[L_seq_test[i-1], L_seq_test[i]] += 1
            A_trans[2, 0], A_trans[0, 2] = 0, 0
            for i in range(A_trans.shape[0]):    
                A_trans[i,:] /= np.sum(A_trans[i,:])
        return A_trans, B_emis
   
 
if '__main__' == __name__:
    
    dir_path = os.path.dirname(os.path.realpath(__file__))

    L_g = os.listdir(str(dir_path)+r"\Data\Left\\")
    L_g = [str(dir_path)+r"\Data\Left\\"+elmt for elmt in L_g]
    L_d = os.listdir(str(dir_path)+r"\Data\Right\\")
    L_d = [str(dir_path)+r"\Data\Right\\"+elmt for elmt in L_d]
    L_c = os.listdir(str(dir_path)+r"\Data\Center\\")
    L_c = [str(dir_path)+r"\Data\Center\\"+elmt for elmt in L_c]


    L_A_f = []
    L_c_f = []
    c_1 = (1, 127/255, 39/255)
    c_2 = (34/255, 177/255, 76/255)
    c_3 = (0, 162/255, 232/255)
    L_A = []
    L_B = []
    c = [(185/255, 122/255, 87/255),
         (136/255, 0, 21/255),
         (237/255, 28/255, 36/255),
         (1, 174/255, 201/255), 
         (1, 127/255, 39/255),
         (181/255, 230/255, 29/255),
         (34/255, 177/255, 76/255),
         (153/255, 217/255, 234/255),
         (0, 162/255, 232/255),
         (63/255, 72/255, 204/255),
         (163/255, 73/255, 164/255),
         (200/255, 191/255, 231/255),
         (195/255, 195/255, 195/255),
         (127/255, 127/255, 127/255),
         (0, 0, 0)]
    L_bins = [0.05*i for i in range(21)]
    for i in range(len(L_g)):
        fn_traitement = traitement_des_donnees(L_g[i])
        A, B = fn_traitement.func_chaine_de_markov()
        L_A.append(A)
        L_A_f.append(A)
        L_c_f.append(c_1)
        L_B.append(B)
    L_A = np.array(L_A)

    fig = plt.figure()
    plt.suptitle(r'$A_{-1} = $ Nuoc Mam - $A_{1} = $ Water')
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for i in range(1,10):
        if i != 3 and i != 7:
            plt.subplot(3,3,i)
            plt.grid(axis='both')
            for j in range(len(L_A),-1,-1):
                plt.hist(L_A[:j, (i-1)//3, (i-1)%3], bins=L_bins, color=c[j-1], ec='k')
            if (i-1)//3 < 2:
                plt.gca().get_xaxis().set_tick_params(labelsize  = 0,size =0)
            else:
                plt.xticks(np.linspace(0,1,6))
            if (i-1)%3 > 0 and i != 8:
                plt.gca().get_yaxis().set_tick_params(labelsize  = 0,size =0)
            plt.xlim(0, 1)
            plt.ylim(0, 10)
    fig.supxlabel("Probability")
    fig.supylabel("Occurence")

    L_A = []
    L_B = []
    for i in range(len(L_c)):
        fn_traitement = traitement_des_donnees(L_c[i])
        A, B = fn_traitement.func_chaine_de_markov()
        L_A.append(A)
        L_B.append(B)
        L_A_f.append(A)
        L_c_f.append(c_2)
    
    L_A = np.array(L_A)

    fig = plt.figure()
    plt.suptitle(r'$A_{-1} = $ Water - $A_{1} = $ Water')
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for i in range(1,10):
        if i != 3 and i != 7:
            plt.subplot(3,3,i)
            plt.grid(axis='both')
            for j in range(len(L_A),-1,-1):
                plt.hist(L_A[:j, (i-1)//3, (i-1)%3], bins=L_bins, color=c[j-1], ec='k')
            if (i-1)//3 < 2:
                plt.gca().get_xaxis().set_tick_params(labelsize  = 0,size =0)
                plt.xticks(np.linspace(0,1,6))
            else:
                plt.xticks(np.linspace(0,1,6))
            if (i-1)%3 > 0 and i != 8:
                plt.gca().get_yaxis().set_tick_params(labelsize  = 0,size =0)
            plt.yticks(np.linspace(0,15,6))
            plt.xlim(0, 1)
            plt.ylim(0, 15)
    fig.supxlabel("Probability")
    fig.supylabel("Occurence")
    
    L_A = []
    L_B = []
    for i in range(len(L_d)):
        fn_traitement = traitement_des_donnees(L_d[i])
        A, B = fn_traitement.func_chaine_de_markov()
        L_A.append(A)
        L_B.append(B)
        L_A_f.append(A)
        L_c_f.append(c_3)
    L_A = np.array(L_A)

    fig = plt.figure()
    plt.suptitle(r'$A_{-1} = $ Water - $A_{1} = $ Nuoc Mam')
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for i in range(1,10):
        if i != 3 and i != 7:
            plt.subplot(3,3,i)
            plt.grid(axis='both')
            for j in range(len(L_A),-1,-1):
                plt.hist(L_A[:j, (i-1)//3, (i-1)%3], bins=L_bins, color=c[j-1], ec='k')
            if (i-1)//3 < 2:
                plt.gca().get_xaxis().set_tick_params(labelsize  = 0,size =0)
            else:
                plt.xticks(np.linspace(0,1,6))
            if (i-1)%3 > 0 and i != 8:
                plt.gca().get_yaxis().set_tick_params(labelsize  = 0,size =0)
            plt.xlim(0, 1)
            plt.ylim(0, 10)
    fig.supxlabel("Probability")
    fig.supylabel("Occurence")



    fig = plt.figure()

    L_A_f = np.array(L_A_f)

    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for i in range(1,10):
        if i != 3 and i != 7:
            plt.subplot(3,3,i)
            plt.grid(axis='both')
            for j in range(len(L_A_f),-1,-1):
                plt.hist(L_A_f[:j, (i-1)//3, (i-1)%3], bins=L_bins, color=L_c_f[j-1], ec='k')
            if (i-1)//3 < 2:
                plt.gca().get_xaxis().set_tick_params(labelsize  = 0,size =0)
            else:
                plt.xticks(np.linspace(0,1,6))
            if (i-1)%3 > 0 and i != 8:
                plt.gca().get_yaxis().set_tick_params(labelsize  = 0,size =0)
            plt.xlim(0, 1)
            plt.ylim(0, 30)
            plt.xticks(np.linspace(0,1,6))
            plt.yticks(np.linspace(0,30,6))
    fig.supxlabel("Probability")
    fig.supylabel("Occurence")

    plt.figure()
    plt.plot([-1], [-1,], color=c_1, label=r'$A_{-1} = $ Nuoc Mam - $A_{1} = $ Water')
    plt.plot([-1], [-1,], color=c_2, label=r'$A_{-1} = $ Water - $A_{1} = $ Water')
    plt.plot([-1], [-1,], color=c_3, label=r'$A_{-1} = $ Water - $A_{1} = $ Nuoc Mam')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.gca().set_xlabel([])
    plt.gca().set_ylabel([])
    plt.axis("off")
    plt.show()