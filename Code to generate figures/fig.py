import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

            if L_temps[i] > 80 and L_temps[i] < 83:
                print(cond)
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
        plt.ylabel('Arm')
        plt.xlabel('Temps (s)')
        if len(self._L_t_max) == 0:
            plt.xlim(0,max(np.max(L_t_aff_choix), np.max(L_t_vit)))
        else:
            plt.xlim(0,self._L_t_max[1])
        plt.grid(axis='x',zorder=0)
        plt.subplots_adjust(hspace=0.02)
        plt.subplot(312)
        plt.plot(L_t_vit, 1000*np.array(L_temps_du_passage))
        plt.ylabel('Maximum passage width (ms)')
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
        plt.hist(L_pixel_0_1 +L_pixel_1_0 +L_pixel_0__1 +L_pixel__1_0, color=(59/255,82/255,139/255), ec='k', bins = L_axe, label=r'Arm 0 $\rightarrow$ Arm -1')
        plt.hist(L_pixel_0_1 +L_pixel_1_0 +L_pixel__1_0, color=(33/255,145/255,140/255), ec='k', bins = L_axe, label=r'Arm -1 $\rightarrow$ Arm 0')
        plt.hist(L_pixel_0_1 +L_pixel_1_0, color=(92/255,200/255,99/255), ec='k', bins = L_axe, label=r'Arm 0 $\rightarrow$ Arm 1')
        plt.hist(L_pixel_1_0, color=(253/255,231/255,37/255), ec='k', bins = L_axe, label=r'Arm 1 $\rightarrow$ Arm 0')    
        plt.ylabel('Occurence')
        plt.xlabel('Pixel')
        plt.legend()
        plt.xlim(0,99)
        plt.gca().twiny()
        plt.xticks(np.linspace(0,22,6))
        plt.xlabel('Size (mm)')

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
        plt.hist(L_bras__1_0 +L_bras_0__1 +L_bras_0_1 +L_bras_1_0, color=(59/255,82/255,139/255), ec='k', bins = L_axe, label=r'Arm 0 $\rightarrow$ Arm -1')
        plt.hist(L_bras_0_1 +L_bras_1_0 +L_bras__1_0, color=(33/255,145/255,140/255), ec='k', bins = L_axe, label=r'Arm -1 $\rightarrow$ Arm 0')
        plt.hist(L_bras_0_1 +L_bras_1_0, color=(92/255,200/255,99/255), ec='k', bins = L_axe, label=r'Arm 0 $\rightarrow$ Arm 1')
        plt.hist(L_bras_1_0, color=(253/255,231/255,37/255), ec='k', bins = L_axe, label=r'Arm 1 $\rightarrow$ Arm 0')    
        plt.ylabel('Occurence')
        plt.xlabel('Maximum passage width (mm)') 
        plt.xlim(0, np.max(L_axe))
        plt.legend()

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
            plt.xlabel('Duration of arm passage (s)')
            plt.ylabel('Occurence')
            plt.xlim(0,1.5*max([np.max([0]+L_temps_bras__1 + L_temps_bras_0 + L_temps_bras_1)]))
            plt.legend()

def fonc_info(lien:str)->bool:
    tab = pd.read_excel(lien, index_col=0)  
    tab = tab.to_numpy()
    L = []
    ind_c = 0

    if tab[0][0][:4] == "nuoc" or tab[0][0][:4] == "Nuoc" \
        or tab[0][0][:4] == "nouc" or tab[0][0][:4] == "Nouc" \
        or tab[0][0][:4] == "muoc" or tab[0][0][:4] == "Muoc":
        ind_c = -1
    elif tab[0][1][:4] == "nuoc" or tab[0][1][:4] == "Nuoc" \
        or tab[0][1][:4] == "nouc" or tab[0][1][:4] == "Nouc" \
        or tab[0][1][:4] == "muoc" or tab[0][1][:4] == "Muoc":
        ind_c = 1
    else:
        ind_c = 0

    tab[1][6] -= 120

    L.append(tab[1][6])
    L_C = []
    L_L = []
    L_R = []
    i = 1
    while tab[i, 2] != 'End':
        if tab[i, 7] == 'C':
            L_C.append(tab[i,6])
        elif tab[i, 7] == 'R':
            L_R.append(tab[i,6])
        else:
            L_L.append(tab[i,6])
        i += 1
    L.append([sum(L_L)/sum(L_L+L_C+L_R), sum(L_C)/sum(L_L+L_C+L_R), sum(L_R)/sum(L_L+L_C+L_R)])
    L.append([L_L, L_C, L_R])

    return ind_c, L

if '__main__' == __name__:
    dir_path = os.path.dirname(os.path.realpath(__file__))

    L_g = os.listdir(str(dir_path)+r"\Data\Left\\")
    L_g = [str(dir_path)+r"\Data\Left\\"+elmt for elmt in L_g]
    L_d = os.listdir(str(dir_path)+r"\Data\Right\\")
    L_d = [str(dir_path)+r"\Data\Right\\"+elmt for elmt in L_d]
    L_c = os.listdir(str(dir_path)+r"\Data\Center\\")
    L_c = [str(dir_path)+r"\Data\Center\\"+elmt for elmt in L_c]

    lien, _ = os.path.split(os.path.realpath(__file__))
    
    # Select an index of gammarus e.g. 0
    fn_traitement = traitement_des_donnees(L_g[0])
    fn_traitement.func_figure_trajectoire()
    fn_traitement.func_figure_lateralisation()
    fn_traitement.func_figure_largeur_passage()
    fn_traitement.func_proba_temps()

    plt.show()