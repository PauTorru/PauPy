
#!/usr/bin/env python





def read_dft_eloss(fname):
    ''' Aquesta merda llegeix arxius .eloss de simulacions DFT i retorna (eaxis, xx elossfunction, zz eloss function)'''   
    e=[]
    lossxx=[]
    losszz=[]
    for line in open(fname).readlines()[7:]:
        #print line.split(' ')
        next_line=False
        for i in range(len(line.split(' '))):
            if line.split(' ')[i]!='' and not next_line:       
                e.append(float(line.split(' ')[i]))
                lossxx.append(float(line.split(' ')[i+2])) 
                losszz.append(float(line.split(' ')[i+4][:-1]))
                next_line=True
        
    return e,lossxx,losszz
    
    
def read_dft_elnes(fname):
    ''' Aquesta merda llegeix arxius .elnes de simulacions DFT i retorna (eaxis, elnes)'''       
    e=[]
    elnes=[]
    
    for line in open(fname).readlines()[11:]:
        fields=line.split('  ')
        if line.split('  ')[0]=='':
            fields.pop(0)
        
        e.append(float(fields[0]))
        elnes.append(float(fields[1]))

    return e,elnes

def read_dft_broadspec(fname):
    ''' Aquesta merda llegeix arxius .eloss de simulacions DFT i retorna (eaxis, total, first, second)'''   
    e=[]
    total=[]
    first=[]
    second=[]
    
    for line in open(fname).readlines()[9:]:
        fields = line.split('  ')
        count=0
        for field in fields:
            try:
                field_int=float(field)
                count+=1
                if count==1:
                    e.append(field_int)
                if count==2:
                    total.append(field_int)
                if count==3:
                    first.append(field_int)
                if count==4:
                    second.append(field_int)
            except:
                pass
            
    return e,total,first,second
            
