import h5py
import numpy as np
import pandas as pd
import math, sys, time

import ROOT as rt

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

h5File = h5py.File('samples/l1jet-pt-regression-h5_SingleMuonRun2018D_0001-3/L1Ntuple_3-1346_3.h5','r')

l1jets = np.array(h5File.get("l1Jet"))
l1jet_pt = l1jets[:,0]
l1jet_eta = l1jets[:,1]
l1jet_phi = l1jets[:,2]

nEvents = l1jet_pt.shape[0]

event = np.array(h5File.get("eventInfo"))
nL1Jets = event[:,4]
evt_HT = event[:,5]
evt_mHT = event[:,7]

nHTBins = 100
HTLo = 0.
HTHi = 1000.
HTBinWidth = (HTHi-HTLo)/nHTBins  

myHT = rt.TH1F("HTRates_emu", "HTRates_emu", nHTBins, HTLo, HTHi)
evtHT = rt.TH1F("HTRates_regression", "HTRates_regression", nHTBins, HTLo, HTHi)

HTx = 0.
HTy = 0.
thisEvent = 0
isSat = False

for e in range(nEvents):
	
        if e%100000 == 0: print("Event",e,"/",nEvents)
        if e < thisEvent+nL1Jets[thisEvent] and (thisEvent+nL1Jets[thisEvent]) <= nEvents:
	
	 myjet = rt.TVector3()
	 myjet.SetPtEtaPhi(l1jet_pt[e],l1jet_eta[e],l1jet_phi[e])
	 
	 if l1jet_pt[e] > 1023: isSat=True
	 
         if round(l1jet_pt[e],1) > 30. and math.fabs(l1jet_eta[e]) < 2.5:
	 	HTx += myjet.Px()
	 	HTy += myjet.Py()
	 
         print(e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent],l1jet_pt[e],l1jet_eta[e],myjet.Px(),myjet.Py(),HTx,HTy)
	 
	 if (thisEvent+nL1Jets[thisEvent]) == nEvents:
	  if not isSat: mHT = math.sqrt(HTx*HTx+HTy*HTy)
	  else: mHT = 2047.5
	  myHT.Fill(mHT)
	  evtHT.Fill(evt_mHT[thisEvent])	  
	 	 
        else:

	 #fill histo and reset
         
	 if not isSat: mHT = math.sqrt(HTx*HTx+HTy*HTy)
	 else: mHT = 2047.5
	 myHT.Fill(mHT)
	 evtHT.Fill(evt_mHT[thisEvent])
	 
	 print "Event",e,"thisEvent",thisEvent,"mHT",mHT,"evtmHT",evt_mHT[thisEvent]
	 #if HT>0 and round(HT,1)!=round(evt_HT[thisEvent],1) and evt_HT[thisEvent]!=2047.5: print "----------------> JEN FOUND YOU NOW !!????",e
	 if mHT > 280 and mHT < 300: print "----------------> JEN FOUND BADMHT??",e
	      
         thisEvent = e
         HTx=0.
	 HTy=0.
	 isSat = False
	 myjet = rt.TVector3()
	 myjet.SetPtEtaPhi(l1jet_pt[e],l1jet_eta[e],l1jet_phi[e])
	 if l1jet_pt[e] > 1023: isSat=True
         if round(l1jet_pt[e],1) > 30. and math.fabs(l1jet_eta[e]) < 2.5:
	 	HTx += myjet.Px()
	 	HTy += myjet.Py()
	 print("--------------------------------------------------------------------")		 
	 print("--------------------------------------------------------------------")		 
         print("::::ELSE::::",e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent],l1jet_pt[e],l1jet_eta[e],myjet.Px(),myjet.Py(),HTx,HTy)

print myHT.GetEntries(),evtHT.GetEntries(),myHT.Integral(),evtHT.Integral()

myHT.SetLineColor(rt.kRed)
myHT.SetMarkerColor(rt.kRed)
myHT.SetMarkerStyle(20)
evtHT.SetLineColor(rt.kBlack)
c = rt.TCanvas()
c.cd()
evtHT.Draw('HIST')
myHT.Draw("PEsame")
time.sleep(1000)
	 
'''
HT = 0.
thisEvent = 0

for e in range(nEvents):

        if e%100000 == 0: print("Event",e,"/",nEvents)
        if e < thisEvent+nL1Jets[thisEvent] and (thisEvent+nL1Jets[thisEvent]) <= nEvents:
	
         if round(l1jet_pt[e],1) > 30. and math.fabs(l1jet_eta[e]) < 2.5: HT+=l1jet_pt[e]
         print(e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent],l1jet_pt[e],l1jet_eta[e],HT)
	 
	 if (thisEvent+nL1Jets[thisEvent]) == nEvents:
	  myHT.Fill(HT)
	  evtHT.Fill(evt_HT[thisEvent])	  
	 	 
        else:

	 #fill histo and reset
         
	 myHT.Fill(HT)
	 evtHT.Fill(evt_HT[thisEvent])
	 #print "Event",e,"thisEvent",thisEvent,"HT",HT,"evtHT",evt_HT[thisEvent]
	 if HT>0 and round(HT,1)!=round(evt_HT[thisEvent],1) and evt_HT[thisEvent]!=2047.5: print "----------------> JEN FOUND YOU NOW !!????",e
	      
         thisEvent = e
         HT=0.
         if round(l1jet_pt[e],1) > 30. and math.fabs(l1jet_eta[e]) < 2.5: HT=l1jet_pt[e]
         #print("::::ELSE::::",e,thisEvent,nL1Jets[thisEvent],thisEvent+nL1Jets[thisEvent],l1jet_pt[e],l1jet_eta[e],HT)

print myHT.GetEntries(),evtHT.GetEntries(),myHT.Integral(),evtHT.Integral()

myHT.SetLineColor(rt.kRed)
myHT.SetMarkerColor(rt.kRed)
myHT.SetMarkerStyle(20)
evtHT.SetLineColor(rt.kBlack)
c = rt.TCanvas()
c.cd()
evtHT.Draw('HIST')
myHT.Draw("PEsame")
time.sleep(1000)	     
'''
	     
'''
h5File = h5py.File('cmssw_dir/test.h5','r')

muons = np.array(h5File.get("recoMuon"))
jets = np.array(h5File.get("recoJet"))
l1jets = np.array(h5File.get("l1Jet"))
event = np.array(h5File.get("eventInfo"))

nvtx = jets[:,4]
nvtx_event = event[:,5]

plt.figure()
plt.hist(nvtx, bins=100, range=[0,100], color='green', histtype='step', label='jets' )
plt.hist(nvtx_event, bins=100, range=[0,100], color='orange', histtype='step', label='events')
plt.legend(loc='upper right')
plt.savefig('nvtx.png')

sys.exit()

print(muons.shape)
print(jets.shape)

muon_pt = muons[:,0]
muon_eta = muons[:,1]
muon_phi = muons[:,2]
jet_pt = jets[:,0]
jet_eta = jets[:,1]
jet_phi = jets[:,2]
dr = muons[:,4]
isomu = muons[:,5]
l1jet_pt = l1jets[:,0]

res = (jet_pt-l1jet_pt)/jet_pt
res_1 = res[np.where(dr > 0.5)]
res_2 = res[np.where(dr < 0.5)]
res_3 = res[np.where(dr < 0.2)]
plt.figure()
plt.hist(res_1, bins=100, range=(-2,1), color='green', histtype='step', label='dr>0.5')
plt.hist(res_2, bins=100, range=(-2,1), color='orange', histtype='step', label='dr<0.5')
plt.hist(res_3, bins=100, range=(-2,1), color='purple', histtype='step', label='dr<0.2')
#plt.hist(jet_pt_6, bins=100, color='deepskyblue', histtype='step', density=True, label='dr>1.0')
plt.hist(res, bins=100, range=(-2,1), color='black', histtype='step', label='inclusive')
plt.legend(loc='upper right')
plt.savefig('res_drs.png')

res_4 = res[np.where(isomu == 1)]
res_5 = res[np.where(isomu == 0)]
plt.figure()
#plt.semilogy()
plt.hist(res_4, bins=100, range=(-2,1), color='green', histtype='step', density=True, label='isomu==1')
plt.hist(res_5, bins=100, range=(-2,1), color='orange', histtype='step', density=True, label='isomu==0')
plt.hist(res, bins=100, range=(-2,1), color='black', histtype='step', density=True, label='inclusive')
plt.legend(loc='upper right')
plt.savefig('res_isomu.png')

#deta = np.fabs(jet_eta - muon_eta)
#dphi = np.fabs(jet_phi - muon_phi)
#dr = np.sqrt(deta**2 + dphi**2)
plt.figure()
plt.semilogy()
plt.hist(dr, bins=100, facecolor='green', alpha=0.75)
plt.savefig('dr.png')

plt.figure()
plt.semilogy()
plt.hist(isomu, bins=2, range=(0,2), facecolor='green', alpha=0.75)
plt.savefig('isomu.png')

jet_pt_1 = jet_pt[np.where(dr > 0.5)]
jet_pt_2 = jet_pt[np.where(dr < 0.5)]
jet_pt_3 = jet_pt[np.where(dr < 0.2)]
#jet_pt_6 = jet_pt[np.where(dr > 2.0)]
plt.figure()
plt.semilogy()
plt.hist(jet_pt_1, bins=100, color='green', histtype='step', density=True, label='dr>0.5')
plt.hist(jet_pt_2, bins=100, color='orange', histtype='step', density=True, label='dr<0.5')
plt.hist(jet_pt_3, bins=100, color='purple', histtype='step', density=True, label='dr<0.2')
#plt.hist(jet_pt_6, bins=100, color='deepskyblue', histtype='step', density=True, label='dr>1.0')
plt.hist(jet_pt, bins=100, color='black', histtype='step', density=True, label='inclusive')
plt.legend(loc='upper right')
plt.savefig('jet_pt_drs.png')

jet_pt_4 = jet_pt[np.where(isomu == 1)]
jet_pt_5 = jet_pt[np.where(isomu == 0)]
plt.figure()
plt.semilogy()
plt.hist(jet_pt_4, bins=100, color='green', histtype='step', density=True, label='isomu==1')
plt.hist(jet_pt_5, bins=100, color='orange', histtype='step', density=True, label='isomu==0')
plt.hist(jet_pt, bins=100, color='black', histtype='step', density=True, label='inclusive')
plt.legend(loc='upper right')
plt.savefig('jet_pt_isomu.png')

jet_pt_6 = jet_pt[np.where(np.fabs(jet_eta) < 2.4)]
jet_pt_7 = jet_pt[np.where(np.fabs(jet_eta) > 2.4)]
plt.figure()
plt.semilogy()
plt.hist(jet_pt_6, bins=100, color='green', histtype='step', density=True, label='|eta| < 2.4')
plt.hist(jet_pt_7, bins=100, color='orange', histtype='step', density=True, label='|eta| > 2.4')
plt.hist(jet_pt, bins=100, color='black', histtype='step', density=True, label='inclusive')
plt.legend(loc='upper right')
plt.savefig('jet_pt_eta.png')

plt.figure()
#plt.semilogy()
plt.hist2d(jet_pt,dr,bins=[40,50],range=[[0, 200], [0, 1]],norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.savefig('hist2d.png')

plt.figure()
#plt.semilogy()
plt.hist2d(res,dr,bins=[100,100],range=[[-2, 1], [0, 5]],norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.savefig('res_hist2d.png')

plt.figure()
plt.semilogy()
plt.hist(muons[:,3], bins=20, range=(0,20), facecolor='green', alpha=0.75)
plt.hist(muons[:,5], bins=20, range=(0,20), facecolor='orange', alpha=0.5)
plt.savefig('m_min.png')
'''

'''
muon_pt = muons[:,0]
print(muon_pt.shape)
print(muon_pt)
print(np.amin(muon_pt))

plt.figure()
plt.semilogy()
plt.hist(muon_pt, bins=100, facecolor='green', alpha=0.75)
plt.savefig('muon_pt.png')

jet_pt = jets[:,0]
print(jet_pt.shape)
print(jet_pt)
print(np.amin(jet_pt))

plt.figure()
plt.semilogy()
plt.hist(jet_pt, bins=100, facecolor='green', alpha=0.75)
plt.savefig('jet_pt.png')

jet_pt = jet_pt[np.where(muons[:,3] == 0)]
print(jet_pt.shape)
print(jet_pt)
print(np.amin(jet_pt))

plt.figure()
plt.semilogy()
plt.hist(jet_pt, bins=100, facecolor='green', alpha=0.75)
plt.savefig('jet_pt_NOhltisomu.png')
'''
