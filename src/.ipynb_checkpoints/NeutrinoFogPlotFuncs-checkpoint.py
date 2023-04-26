#===========================NeutrinoFogPlotFuncs.py===================================#
# Created by Bing-Long Zhang, 2023

# Contains functions for making some plots.
# All functions below are developed by modifying Ciaran's Python code.

#==============================================================================#
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Params import *
from NeutrinoFuncs import *
from PlotFuncs import *
from Like import *
import cmasher as cmr
from matplotlib.colors import LinearSegmentedColormap
from NeutrinoFogFuncs import *

from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,mark_inset)
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from numpy import *

# The function for reproducing the neutrino fog as the Figure. 2 in 2109.03116.
def vFogPlot1(res, m_vals, R_nu):
    interval = linspace(0.12,0.9)
    colors = cmr.pride(interval)
    cmap = LinearSegmentedColormap.from_list('name', colors)

    data = loadtxt('../data/WIMPLimits/mylimits/DLNuFloorXe_detailed_SI.txt')

    m, sig, DY = m_vals, np.logspace(-50,-39,500), plotDatGen(m_vals, res)[2]
    NUFLOOR = np.array([m_vals, np.transpose(np.array(list(map(findPoint, res))))[1]])

    R = np.sum(R_nu[5])

    pek = line_background(10,'k')
    #cmap = cmr.pride  
    nmax = 13
    vmax = 11
    vmin = 2
    m_example = 5.5

    # Figure setup
    import matplotlib.gridspec as gridspec
    lw = 2.5
    lfs = 35
    tfs = 25
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
    fig = plt.figure(figsize=(20,15))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.04)
    gs.update(hspace=0.04)
    ax1 = plt.subplot(gs[:,0:2])
    ax2 = plt.subplot(gs[0,-1])
    ax3 = plt.subplot(gs[1,-1])

    # Plot limits
    SILimits(ax1,Annotations=False)


    # Plot neutrino fog
    col_min = cmap(0.0)
    cnt = ax1.contourf(m,sig,DY,levels=linspace(2,15,100),vmin=2.3,vmax=vmax,cmap=cmap,zorder=-100)
    for c in cnt.collections: c.set_edgecolor("face")
    ax1.plot(NUFLOOR[0],NUFLOOR[1],'-',color='gray',lw=3,path_effects=pek,zorder=1)
    ax1.fill_between(NUFLOOR[0],NUFLOOR[1],y2=1e-99,color=col_min,zorder=-1000)


    # Plot single mass example line
    x1 = m_example
    x2 = m_example
    y1 = 1e-44
    y2 = 6e-47

    # Calculate derivative of discovery limit
    i = np.argmin(abs(m-m_example))
    E = np.flip(res[i][0]*R)
    def ydyGen(dat):
        x0 = dat[0]*R
        y0 = 10**(np.array(list(map(lambda i: myFindRoot(dat[2][i]), range(len(dat[2]))))))
        x1 = 1/2*(x0[1:]+x0[:-1])
        y1 = -(x1/(1/2*(y0[1:]+y0[:-1]))*(y0[1:]-y0[:-1])/(x0[1:]-x0[:-1]))**(-1)
        yp = 1/2*(y0[1:]+y0[:-1])
        return [np.flip(y0), np.flip(yp), np.flip(y1)]
    y, yp, dy =  ydyGen(res[i])
    Ep = 1/2*(E[1:]+E[0:-1])

    ax1.plot([x1,x1],[y1,y2],'w-',lw=3,path_effects=pek)
    ax1.plot(x1,y1,'wo',markersize=30,mfc='w',mec='k',mew=3)
    ax1.plot(x2,y2,'wo',markersize=30,mfc='w',mec='k',mew=3)
    ax1.plot(x2,y2,'wo',markersize=30,mfc='w',mec='k',mew=3)

    # Plot white circles (ax2)
    ax2.plot(np.interp(y1,y,E),y1,'wo',markersize=30,mfc='w',mec='k',mew=3,zorder=10)
    ax2.plot(np.interp(y2,y,E),y2,'wo',markersize=30,mfc='w',mec='k',mew=3,zorder=10)
    ax3.plot(np.interp(y1,y,E),np.interp(y1,yp,dy),'wo',markersize=30,mfc='w',mec='k',mew=3,zorder=10)
    ax3.plot(np.interp(y2,y,E),np.interp(y2,yp,dy),'wo',markersize=30,mfc='w',mec='k',mew=3,zorder=10)

    # Poisson line
    sigfloor = yp[dy>2][-1]
    E1 = np.interp(sigfloor,y,E)
    sig_poiss = sigfloor*np.sqrt(E1)/np.sqrt(E)
    ax2.plot(E,sig_poiss,'k:',zorder=1,alpha=0.5)
    ax2.text(6e3,0.45e-47,r'$\sigma \propto N^{-1/2}$',color='k',alpha=0.5,rotation=-46)

    # Plot horizontal/vertical lines
    Efloor = Ep[dy>2][-1]
    ax2.axhline(sigfloor,color='gray',lw=3,path_effects=pek)
    ax2.axvline(Efloor,lw=3,color='k',linestyle='--',alpha=0.5)
    ax3.axvline(Efloor,lw=3,color='gray',path_effects=pek)
    ax3.axhline(2,color='k',linestyle='--',lw=3,alpha=0.5)


    # Plot coloured lines
    ax2.loglog(E,y,lw=3,color='w',path_effects=pek)
    ax3.semilogx(Ep,dy,lw=3,color='w',path_effects=pek)
    cval = (((dy - vmin)) / (vmax-vmin))
    cval[isnan(cval)] = 0
    cval[cval<0] = 0
    cval[cval>1] = 1
    for i in range(0,size(dy)-1):
        ax2.loglog([E[i],E[i+1]],[y[i],y[i+1]],color=cmap(cval[i]),lw=5,solid_capstyle='round')
        ax3.semilogx([Ep[i],Ep[i+1]],[dy[i],dy[i+1]],color=cmap(cval[i]),lw=5,solid_capstyle='round')


    # Neutrino fog
    for i in logspace(-7,0,100):
        ax2.fill_between(E,ones((size(E)))*sigfloor*i,y2=1e-99,color='gray',alpha=0.05,lw=0)
    for i in logspace(0,12,100):
        ax3.fill_between([Efloor*i,1e10],[15,15],y2=-15,color='gray',alpha=0.05,lw=0)


    # Text inside white circles
    ax1.text(x1*(1+0.01),y1*(1-0.01),'a',fontsize=23,ha='center',va='center')
    ax1.text(x2,y2*(1-0.05),'b',fontsize=23,ha='center',va='center')
    ax2.text(np.interp(y1,y,E)*(1+0.02),y1*(1-0.01),'a',fontsize=23,ha='center',va='center',zorder=10)
    ax2.text(np.interp(y2,y,E),y2*(1-0.05),'b',fontsize=23,ha='center',va='center',zorder=10)
    ax3.text(np.interp(y1,y,E)*(1+0.02),np.interp(y1,yp,dy)*(1-0.01),'a',fontsize=23,ha='center',va='center',zorder=10)
    ax3.text(np.interp(y2,y,E),np.interp(y2,yp,dy)*(1-0.02),'b',fontsize=23,ha='center',va='center',zorder=10)



    # Axis limits
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax1.set_xlim([1e-1,1e4])
    ax1.set_ylim([1e-50,1e-40])
    ax2.set_ylim([1e-48,1e-43])
    ax2.set_xlim([2e-1,1e9])
    ax3.set_xlim([2e-1,1e9])
    ax3.set_ylim([0.9,nmax])
    ax2.set_xticklabels([])


    # Text labels
    ax1.set_xlabel(r"DM Mass [GeV$/c^2$]",fontsize=lfs)
    ax1.set_ylabel(r"SI DM-nucleon cross section [cm$^2$]",fontsize=lfs)
    ax2.set_ylabel(r'$\sigma$ [cm$^2$]',fontsize=lfs,rotation=-90,labelpad=45)
    ax3.set_ylabel(r'$n$',fontsize=lfs,rotation=-90,labelpad=30)
    ax3.set_xlabel(r'Number of $^8$B events',fontsize=lfs)
    ax2.text(1e6,sigfloor*1.5,r'{\bf Neutrino ``floor"}',color='k',ha='center')
    ax2.text(1e6,sigfloor/2.5,r'{\bf Neutrino fog}',color='k',alpha=0.5,ha='center')
    plt.gcf().text(0.16*(1-0.01),0.17*(1+0.01),r'{\bf Xenon}',color='k',fontsize=50,alpha=0.2)
    plt.gcf().text(0.16,0.17,r'{\bf Xenon}',color='w',fontsize=50)


    # Limits labels
    ax1.text(0.55,2.4e-41,r"{\bf CDMSlite}",color="blue",fontsize=22,rotation=0)
    ax1.text(2.7,3e-42,r"{\bf DarkSide}",color="green",fontsize=22,rotation=0,ha='right')
    ax1.text(850.0,2.e-45,r"{\bf PandaX}",color="navy",fontsize=22,rotation=25)
    ax1.text(2000.0,0.9e-45,r"{\bf XENON1T}",color="darkgreen",fontsize=22,rotation=27)
    ax1.text(7.0,2.9e-44,r"{\bf EDELWEISS}",color=[0.67, 0.31, 0.32],fontsize=18,rotation=-30)
    ax1.text(2000.0,5.4e-43,r"{\bf PICO60}",color=[0.5, 0.0, 0.13],fontsize=22,rotation=25)
    ax1.text(500.0,1.5e-41,r"{\bf PICO2L}",color=[0.5, 0.0, 0.13],fontsize=22,rotation=25)
    ax1.text(30.0,1.1e-41,r"{\bf DAMA}",color='darkslategray',fontsize=22,rotation=41)
    ax1.text(1100.0,1e-41,r"{\bf COSINE-100}",color="gold",fontsize=22,rotation=25)
    ax1.text(1.5e3,3.5e-44,r'{\bf DEAP-3600}',color='#4ff09d',fontsize=22,rotation=25)
    ax1.text(4000.0,8.7e-45,r"{\bf LUX}",color="crimson",fontsize=21,rotation=25.5)
    CurvedArrow(2.6e3,3.8e3,0.6e-44,0.45e-44,alpha=1,color='navy',connectionstyle="arc3,rad=-0.3",\
                style = "Simple, tail_width=2, head_width=6, head_length=8")

    # Neutrino labels
    ax1.text(1e3,5e-49,'Atmospheric',color='w',alpha=0.85,fontsize=22,rotation=25)
    ax1.text(30,2e-50,'DSNB',color='w',alpha=0.85,fontsize=22,rotation=0)

    ax1.text(4.5,4e-48,r'$hep$',color='w',alpha=0.85,fontsize=22,rotation=0)

    ax1.text(3.5,4.5e-46,r'$^8$B',color='w',alpha=0.99,fontsize=22,rotation=0)

    ax1.text(0.8,5e-48,r'Reactor',color='w',alpha=0.99,fontsize=18,rotation=0)

    ax1.text(0.6,2e-47,r'Geo',color='w',alpha=0.85,fontsize=20,rotation=0)

    ax1.text(0.45,7e-46,r'CNO',color='w',alpha=0.9,fontsize=20,rotation=0)

    ax1.text(0.39,6e-45,r'$pep$',color='w',alpha=0.99,fontsize=20,rotation=0)

    ax1.text(0.28,1.8e-44,r'$^7$Be',color='w',alpha=0.99,fontsize=20,rotation=0)

    ax1.text(0.15,3e-45,r'$^7$Be',color='w',alpha=0.99,fontsize=20,rotation=0)

    ax1.text(0.11,1.5e-46,r'$pp$',color='w',alpha=0.99,fontsize=20,rotation=0)

    # Ticks
    ax1.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True,pad=10)
    ax1.tick_params(which='minor',direction='in',width=1,length=10,right=True,top=True)
    ax2.tick_params(which='major',direction='in',width=2,length=13,left=True,top=True,pad=10)
    ax2.tick_params(which='minor',direction='in',width=1,length=10,left=True,top=True)
    ax3.tick_params(which='major',direction='in',width=2,length=13,left=True,top=True,pad=10)
    ax3.tick_params(which='minor',direction='in',width=1,length=10,left=True,top=True)
    ax1.set_yticks(10.0**arange(-50,-39,1))

    # Colorbar
    im = ax1.pcolormesh(-m,sig,DY,vmin=vmin,vmax=vmax,cmap=cmap,rasterized=True,shading='auto')
    fig.subplots_adjust(top=0.8)
    cbar_ax = fig.add_axes([0.1, 0.82, 0.82, 0.02])
    fig.colorbar(im, cax=cbar_ax,orientation='horizontal',extend='both',extendfrac=0.03)
    cbar_ax.xaxis.set_ticks_position('top')
    plt.gcf().text(0.5,0.88,r'Gradient of discovery limit, $n = -({\rm d}\ln\sigma/{\rm d}\ln N)^{-1}$',fontsize=35,ha='center')
    cbar_ax.tick_params(which='major',direction='in',width=2,length=13,right=True,top=True)
    return fig

# The function for reproducing the neutrino fog as the Figure. 1 in Snowmass 2203.08084.
def vFogPlot2(res, m_vals, color, target):
    #
    n = 5
    colors = np.zeros([n,4])
    colors[0] = mpl.colors.to_rgba(color)
    colors[:] = mpl.colors.to_rgba(color)
    colors[:,3] = [0., 0.2, 0.4, 0.6, 0.8]
    cmap = mpl.colors.ListedColormap(colors, name='test')
    #
    #
    fig,ax = MakeLimitPlot_SI2023(Annotations=True,Collected=False,\
                     xmin=1.0e-1,xmax=1.0e4,ymin=1e-50,ymax=1e-41,\
                     facecolor='darkgray',edgecolor='darkgray',edgecolor_collected='darkgray',\
                     alph=0.5,lfs=35,tfs=25,\
                     xlab=r"DM mass [GeV]",ylab=r"SI DM-nucleon cross section [cm$^2$]")
    m,sig,DY = plotDatGen2(m_vals, res)
    cnt = plt.contourf(m,sig,DY,np.array([1.5, 2., 2.5, 3., 3.5]),\
                       cmap=cmap, vmin=1. , vmax=3.5 )
    for c in cnt.collections:
        c.set_edgecolor("face")
    
    pek = line_background(10,'k')
    NUFLOOR = np.array([m_vals, np.transpose(np.array(list(map(findPoint, res))))[1]])
    [edgeX, edgeY] = NUFLOOR
    plt.plot(edgeX,edgeY,'-',color=color,lw=3,path_effects=pek,zorder=100)
    
    cmap2 = mpl.colors.ListedColormap(colors[1:-1], name='test')
    cmap2.set_under(color=color, alpha=0.)
    cmap2.set_over(color=color, alpha=1.)
    im = plt.pcolormesh(-m,sig,DY,vmax=3.,vmin=1.5,cmap=cmap2,rasterized=True,shading='auto')
    fig.subplots_adjust(top=0.8)
    cbar_ax = fig.add_axes([0.1, 0.82, 0.82, 0.02])
    fig.colorbar(im, cax=cbar_ax,orientation='horizontal',extend='both',extendfrac=0.03, ticks=[1.5,2,2.5,3.])
    cbar_ax.xaxis.set_ticks_position('top')
    plt.gcf().text(0.5,0.883,r'Gradient of discovery limit, $n = -({\rm d}\ln\sigma_{\rm DL}/{\rm d}\ln N)^{-1}$',fontsize=22,ha='center')

    #plt.gcf().text(0.82,0.9,r'$\left(\frac{{\rm d}\ln\sigma}{{\rm d}\ln N}\right)^{-1}$',fontsize=35,color='w')
    plt.gcf().text(0.15*(1-0.01),0.16*(1+0.01),r'\bf {0} neutrino fog'.format(target),color='k',fontsize=30,alpha=0.2)
    plt.gcf().text(0.15,0.16,r'\bf {0} neutrino fog'.format(target),color='black',fontsize=30)
    return fig

# The function for ploting the sensitivity curve in our paper.
def SensitivityCurvePlot(res, Nuc, R_nu, uncList):
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=30)
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'

    m_chi = 5.5
    name = 'Xenon'

    ###### Load in the data
    sig = np.array(list(map(lambda x: 10**(np.array(list(map(lambda i: myFindRoot(x[2][i]), range(len(x[2])))))), res)))
    ferrs = np.array(uncList)*100
    ex_vals = res[0][0]
    n_ex = len(ex_vals)
    nerrs = len(ferrs)  

    N = ex_vals*np.sum(R_nu[5])
    
    col = cm.rainbow(linspace(0,1,nerrs))

    # Set up subplots
    fig, ax = plt.subplots(figsize=(13,13))

    ylab = r"Discovery Limit, $\sigma_{\rm DL}$ [cm$^2$]"

    ymin = 1.1e-49
    ymax = 1.0e-43

    # Labels
    ax.set_ylabel(ylab,fontsize=40)
    ax.set_xlim([N[0],N[-1]])

    # axis
    ax.set_xlabel(r'Exposure [ton-year]',fontsize=40,labelpad=20)  # we already handled the x-label with ax1
    ax.set_xlim([ex_vals[0],ex_vals[-1]])
    ax.set_ylim([ymin,ymax])
    # plot
    zo = 0
    for i in flipud(range(0,nerrs)):
        ax.loglog(ex_vals,sig[i,:],'-',linewidth=6,color=col[i,:],label=str(int(ferrs[i]))+r' \%',zorder=zo)

        dEx = gradient(log10(sig[i,:]))/gradient(log10(N))
        i1 = argmax(dEx)
        sig_dphi = sqrt((1+N*(ferrs[i]/100)**2.0)/N)
        ax.loglog(ex_vals[i1:],sig_dphi[i1:]*(sig[i,i1]/sig_dphi[i1]),'--',linewidth=3,color=col[i,:],zorder=zo)
        zo = zo-1
        if i == nerrs-1:
            plt.text(ex_vals[-30],1.3*sig_dphi[-30]*(sig[i,i1]/sig_dphi[i1]),r'$\propto \sigma_\nu$',fontsize=40,color=col[-1,:],horizontalalignment='right')
    plt.gcf().text(0.80,0.80,r'{\bf '+ name+'}',fontsize=47,horizontalalignment='right')
    plt.gcf().text(0.84,0.72,r'$m_\chi = $ '+'5.5 GeV',fontsize=40,horizontalalignment='right')

    Ex0 = ex_vals[0]
    sig0 = sig[i,0]
    sig_half = (sig0/10.)*(ex_vals/Ex0)**-0.5
    sig_1 = sig0*(ex_vals/Ex0)**-1.0

    ax.loglog(ex_vals[sig_half<sig_1],sig_1[sig_half<sig_1],'k--',linewidth=2.5)
    #ax2.loglog(ex_vals[0:25],sig_1[0:25],'k--',linewidth=2.5)
    ax.text(ex_vals[5],sig_1[1]/5,r'$\propto N^{-1}$',fontsize=38,rotation=-55)
    ax.loglog(ex_vals[60:],sig_half[60:]*7.0e1*2,'k--',linewidth=2.5)
    plt.text(ex_vals[-30],sig_half[-30]/0.005,r'$\propto N^{-1/2}$',fontsize=38,rotation=-45,verticalalignment='top',horizontalalignment='left')
    plt.text(ex_vals[-50],sig_half[-50]/0.012,'Systematical Limit ',fontsize=30,rotation=-45,verticalalignment='top',horizontalalignment='left')
    ax.loglog(ex_vals,sig_half,'k--',linewidth=2.5)
    plt.text(ex_vals[5],sig_half[2]/5,r'$\propto N^{-1/2}$',fontsize=38,rotation=-45,verticalalignment='top',horizontalalignment='left')
    plt.text(ex_vals[27],sig_half[2]/38,'Statistical Limit',fontsize=30,rotation=-45,verticalalignment='top',horizontalalignment='left')


    # Style
    #ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(which='major',direction='in',width=2,length=13,bottom=True,top=False,pad=10)
    ax.tick_params(which='minor',direction='in',width=1,length=10,bottom=True,top=False)

    # legend
    leglab = r'$\sigma_\nu/\sigma_{\nu 0}$'
    leg1 = ax.legend(fontsize=35,frameon=True,loc="lower left",framealpha=1,title=leglab,labelspacing=0.1)
    leg1.get_frame().set_linewidth(3)
    leg1.get_frame().set_edgecolor("k")

    plt.show()
    return fig

