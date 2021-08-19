import os, sys, commands
import optparse
import ROOT

def ResubmitJob(file,dirList):

    for d in dirList:
    
      f = open(d+'/job.sh','r')
      
      for l in f.readlines():
          if '.root' in l:
              if l.split('/')[-1].replace('\n','') == file:
	          os.chdir(d)
	          os.system("condor_submit submit.sub")
	          jobid = d.split('-')[-1]
	          print "Job",jobid,"submitted!"
	          os.chdir('../')
	          break
       

def makeSubmitFileCondor(exe,jobname,jobflavour,localinput=False,cmst3=False):
    #print "make options file for condor job submission "
    submitfile = open("submit.sub","w")        
    submitfile.write("should_transfer_files = YES\n")
    submitfile.write("when_to_transfer_output = ON_EXIT\n")
    submitfile.write('transfer_output_files = ""\n')
    submitfile.write("executable  = "+exe+"\n")
    
    if localinput:
      submitfile.write("arguments             = $(ClusterID) $(ProcId)\n")
    else:
     submitfile.write("Proxy_filename = x509up_%s\n"%os.getenv("USER"))
     submitfile.write("Proxy_path = %s/$(Proxy_filename)\n"%os.getenv("HOME"))
     submitfile.write("transfer_input_files = $(Proxy_path)\n")
     submitfile.write("arguments             = $(Proxy_path) $(ClusterID) $(ProcId)\n")  
    
    submitfile.write("output                = "+jobname+".$(ClusterId).$(ProcId).out\n")
    submitfile.write("error                 = "+jobname+".$(ClusterId).$(ProcId).err\n")
    submitfile.write("log                   = "+jobname+".$(ClusterId).log\n")
    submitfile.write('+JobFlavour           = "'+jobflavour+'"\n')
    if cmst3:
     submitfile.write("+AccountingGroup = group_u_CMST3.all\n")
    submitfile.write("queue")
    submitfile.close()  


if __name__ == "__main__":

        parser = optparse.OptionParser()
	parser.add_option("-l","--label",dest="label",help="Label for local job log directories",default='')
	parser.add_option("-o","--outdir",dest="outdir",help="Output folder name",default='./')
	parser.add_option("-i","--indir" ,dest="indir" ,help="Input folder name" ,default='L1Ntuple_dir')
	parser.add_option("-q","--queue" ,dest="queue" ,help="Queue" ,default='tomorrow')
        parser.add_option("--check_jobs","--check_jobs",dest="check_jobs", action="store_true", help="Check and resubmit failed jobs",default=False)
        parser.add_option("--max","--max",dest="max", type=int, help="Max entries per job",default=-1)
        parser.add_option("--calo","--calo",dest="caloJets", action="store_true", help="Use calo jets",default=False)
        parser.add_option("--h5type","--h5type",dest="h5type",type=int,help="Which h5 type (efficiency/training versus rates)",default=0)    
	(options,args) = parser.parse_args()
	
	indir = options.indir
	outdir = options.outdir
	workdir = os.getcwd()
	
	print
	print "Input dir:",indir
	print "Output dir:",outdir
	print "Label:",options.label
	print "Queue:",options.queue
	print "Max events per job:",options.max
	print "Use calo jets:",options.caloJets
	print "H5 type",options.h5type
	print
	
	###########################################################################################
	if options.check_jobs:
	
	 	import glob
	 	if options.label != '': jobLogDirs = glob.glob('./convertToh5-{label}*[0-9]'.format(label=options.label))
	 	else: jobLogDirs = glob.glob('./convertToh5-*[0-9]')
	 	jobOutFiles = glob.glob(outdir+'/*.h5')
	 	nFailedJobs = len(jobLogDirs)-len(jobOutFiles)
	 	print len(jobLogDirs),"jobs were submitted - found",len(jobOutFiles),"job output files -",nFailedJobs," jobs failed"
	 
	 	if nFailedJobs==0:	 
	  		print "No jobs to resubmit!"
	  		sys.exit()

	 	jobInFiles = glob.glob(indir+"/*.root")	  
	 	for f in jobInFiles:
	  		found = False
	  		for ff in jobOutFiles:
	   			if ff.split('/')[-1].replace('.h5','') == f.split('/')[-1].replace('.root',''):
	    				found = True
	    				break
	 		if found==False:
	   			print "File",f,"not found - resubmitting job ... "
	   			#ResubmitJob(f.split('/')[-1].replace('.root','.h5'),jobLogDirs)
	 	 
		sys.exit()

	###########################################################################################	
	if not os.path.exists(outdir):
	 print "**** The output directory",outdir,"does NOT exist. Creating it..."
	 os.mkdir(outdir)
	else: print "**** The output directory",outdir,"ALREADY exist. Will overwrite the content!"
	print 
		
	files = os.listdir(indir)
	files = []
	for f in os.listdir(indir):
	 if '_3-' in f: files.append(f)
	 #if '_1-' in f or '_2-' in f or '_3-' in f: continue
	 #files.append(f)
	 
	N_entries = []
	max_entries = options.max
	njobs = 0
	for f in files:
	 #if 'L1Ntuple_103.root' in f:
	 # N_entries.append(0)
	 # njobs+=1
	 # continue
	 inf = ROOT.TFile.Open(indir+"/"+f)
	 print f,inf.IsOpen()
	 l1EventTree = inf.Get('l1EventTree/L1EventTree')
	 N_entries.append(l1EventTree.GetEntries())
	 if max_entries != -1:
	  njobs += l1EventTree.GetEntries()/max_entries
	  if l1EventTree.GetEntries()%max_entries != 0: njobs+=1
	 else: njobs+=1 
	 inf.Close()
	
	raw_input("Will submit %i jobs.Press enter to continue..."%njobs)
	#sys.exit()
	
	j = 0
	for i,f in enumerate(files):
 				
                njobs = 1
                if max_entries != -1:
		 njobs = N_entries[i]/max_entries
		 if N_entries[i]%max_entries != 0: njobs+=1

                if j > 1000:
		 print i,f,j,njobs
		 break
		
		#if i <= 167:
		# print i,f,j,njobs
		# continue	

                #if j > 1000:
		# print i,f,j,njobs
		# break

		#if i <= 335:
		# print i,f,j,njobs
		# continue	

                #if j > 1000:
		# print i,f,j,njobs
		# break
		 				 	 		
  		for job in range(njobs):
					 
 			if options.label != '': jobdir = 'convertToh5-'+options.label+'-'+str(j+1)
 			else: jobdir = 'convertToh5-'+str(j+1)
 			if os.path.exists(jobdir):
   				print "Job directory",jobdir,"already exists. Removing it ..."
   				os.system('rm -rf %s'%jobdir)  
 			os.mkdir(jobdir)  
 			os.chdir(jobdir)
			
			if job==0: first_entry = 0
			else: first_entry = job*max_entries + 1
			last_entry = (job+1)*max_entries
			if job == njobs-1: last_entry = N_entries[i]
  
 			infile = indir+"/"+files[i] 
 			outfile = outdir+"/"+files[i].replace('.root','_%i.h5'%(job+1))
 			cmssw_cmd = 'python {workdir}/convertToh5.py -i {infile} -o {outfile} --first {first} --last {last} --h5type {h5type}'.format(workdir=workdir,infile=infile,outfile=outfile,first=first_entry,last=last_entry,h5type=options.h5type)
                        if options.caloJets: cmssw_cmd+=" --calo"
			#print cmssw_cmd

 			with open('job.sh', 'w') as fout:
     				fout.write("#!/bin/sh\n")
     				fout.write("echo\n")
     				fout.write("echo\n")
     				fout.write("echo 'START---------------'\n")
     				fout.write("echo 'WORKDIR ' ${PWD}\n")
     				fout.write("source /afs/cern.ch/cms/cmsset_default.sh\n")
     				fout.write("cd "+str(workdir)+"\n")
     				fout.write("cmsenv\n")
     				fout.write("export X509_USER_PROXY=$1\n")
     				fout.write("echo $X509_USER_PROXY\n")
     				fout.write("voms-proxy-info -all\n")
     				fout.write("voms-proxy-info -all -file $1\n")
     				fout.write("%s\n"%(cmssw_cmd)) 
     				fout.write("echo 'STOP---------------'\n")
     				fout.write("echo\n")
     				fout.write("echo\n")
 				os.system("chmod 755 job.sh")    
   
 			###### sends bjobs ######
 			makeSubmitFileCondor("job.sh","job",options.queue)
 			os.system("condor_submit submit.sub")
 			print "job nr " + str(j+1) + " submitted"
                        j+=1
			
 			os.chdir("../")
   
