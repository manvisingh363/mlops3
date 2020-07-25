final = open('/root/task/last.txt','r')
initial = open('/root/task/initial.txt','r')
accuracy = open('/root/task/acc.txt','r')

#Read the final.txt file and splits it according to requirement as below.
d = last.read()
d = d.split('\n')

old_a =float(d[0])  #initially i'l take old_accuracy that is stored in line1.
layer =int(d[1])  #this values shows that 1 for Convolve layer and 2 for FCLayer.
line =int(d[2])  #indicates the line number in which changes occur.
cp_line =line % 3  #in case of convolve layer(line%3) giving output as 0 for pools,1 for NoOfFilters,2 for strides.
entered_data =int(d[3])  #indicates the changed data which was taken by program as initially input.  
old_data =int(d[4])  #this helps in changing the enetered_data will begin for thr first time in the layer.
index_fc =int(d[5])  #line number of the initial input which shall specify the no. of FCLayer in the program.

new_a =float(accuracy.read())

#NOTE:Firstly the old_data and entered_data will be same.

i = initial.read()
i = i.split('\n')

if new_a>old_a and new_a-old_a>=.00001 :
	old_a=new_a
	if layer == 1:
    		if cp_line == 1:
      			entered_data=entered_data*2
    		else :
      			entered_data+=1
	else:
   		entered_data+=100
	i[line] = str(entered_data)
else:
	if layer == 1:
		if cp_line == 1:
			if entered_data//2 == old_data:
				i=i[0:line]
				i.append('1')
				layer = 2
				index_fc = line				
				i.append('100')
				old_data = 100		
				entered_data = 100
				line = line + 1
				i[0] = str(int(i[0])-1)
			else:
				i[line] = str(entered_data//2)
				line = line+1
				entered_data = 3
				old_data = 2
				i[line] = str(entered_data)
		elif cp_line ==2:
			i[line] = str(entered_data-1)
			line = line+1
			entered_data = 3
			old_data = 2
			i[line] = str(entered_data)
		elif cp_line ==0:
			i[line] = str(entered_data -1)
			line = line+1
			old_data = int(i[line - 3])
			enetered_data =old_data*2
			i[0]=str(int(i[0])+1)
			i=i[0:line]
			i.append(str(entered_data))
			i.append('2')
			i.append('2')
			i.append('0')
			index_fc =line+3
	else:
		nol=int(input[index_fc])+1
		i[index_fc]=str(nol)
		entered_data -=100
		old_data=entered_data
		i[line] =str(entered_data)
		line+=1
		i.append(str(entered_data))
last.close()
initial.close()	

last=open('/root/task/last.txt','w')
initial=open('/root/task/initial.txt','w')

data_file_data = str(old_a) + '\n' + str(layer) + '\n' + str(line) + '\n' + str(entered_data) + '\n' + str(old_data) + '\n' + str(index_fc)			
last.write(data_file_data)
last.close()
initial_file_data='\n'.join(i)
initial.write(initial_file_data)
initial.close()
