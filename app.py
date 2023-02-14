import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
#import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from scipy.stats.qmc import Halton,Sobol
from scipy.stats import norm
from typing import Callable

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

load_figure_template("solar")

class Twiss:
    def __init__(self, alpha: float, beta: float, emit: float):
        self.alpha = alpha
        self.beta = beta
        self.emit = emit

    def __str__(self):
        return f"alpha={self.alpha}\nbeta={self.beta}\nemit={self.emit}"

    def rms_space(self):
        return np.sqrt(self.emit*self.beta)
    
    def rms_angle_waist(self):
        return np.sqrt(self.emit/self.beta)
    
    def gamma(self):
        return (1+self.alpha**2)/self.beta

def ellipse_twiss(twiss_paramters: Twiss, num_points: int = 1000):
    # rmsX = np.sqrt(twiss_paramters.emit/twiss_paramters.beta);
    # rmsTheta = np.sqrt(twiss_paramters.emit*twiss_paramters.beta);
    rmsX = twiss_paramters.rms_space();
    rmsTheta = twiss_paramters.rms_angle_waist();
    
    m = -twiss_paramters.alpha/twiss_paramters.beta
    b = rmsTheta
    a = rmsX
    t = np.linspace(0,2*np.pi,num_points)
    x = a*np.cos(t)
    y = b*np.sin(t)
    x = x.transpose()
    y = y.transpose()
    y = y + x*m
    
    return x,y

def gen_halton_gaussian_4d(twiss_x,twiss_y,num,seed=1,dim=4,loc=[0,0,0,0]):
    rmsX = twiss_x.rms_space();
    rmsThetaX = twiss_x.rms_angle_waist();
    rmsY = twiss_y.rms_space();
    rmsThetaY = twiss_y.rms_angle_waist();    
    
    a=Halton(d=dim,seed=seed)
    vals = a.random(n=num)
    
    dist = norm.ppf(vals,loc=loc,scale=[rmsX,rmsThetaX,rmsY,rmsThetaY])

    mx = -twiss_x.alpha/twiss_x.beta
    my = -twiss_y.alpha/twiss_y.beta;
    
    dist[:,1] += dist[:,0]*mx
    dist[:,3] += dist[:,2]*my
    
    return dist

def gen_dist_normal_coord_4d(func: Callable,**kwarg):
    twiss_x = Twiss(alpha=0,beta=1,emit=1)
    twiss_y = Twiss(alpha=0,beta=1,emit=1)
    return func(twiss_x,twiss_y,**kwarg)

def transform_norm_dist(a: np.ndarray,twiss_x: Twiss,twiss_y: Twiss):
       
    a_copy = a.copy()
    
    twiss_old  = Twiss(alpha=0,beta=1,emit=1)
    
    x_scale = twiss_x.rms_space()/twiss_old.rms_space();
    x_theta_scale = twiss_x.rms_angle_waist()/twiss_old.rms_angle_waist();
    y_scale = twiss_y.rms_space()/twiss_old.rms_space();
    y_theta_scale = twiss_y.rms_angle_waist()/twiss_old.rms_angle_waist();  
    
    mx = -twiss_x.alpha/twiss_x.beta
    my = -twiss_y.alpha/twiss_y.beta;
    
    a_copy[:,0] *= x_scale
    a_copy[:,1] *= x_theta_scale
    a_copy[:,2] *= y_scale
    a_copy[:,3] *= y_theta_scale
    
    a_copy[:,1] += a_copy[:,0]*mx
    a_copy[:,3] += a_copy[:,2]*my
    
    return a_copy

def KE2gamma(KE):
    """Computes the Lorentz Parameter from the kinetic energy
        
    Parameters
    ----------
    KE : array_like
        Kinetic Energy
    Returns
    -------
    array_like
        Lorentz Parameter
    """
    
    mc2 = 0.5109989461e6
    return (1+KE/mc2)
    
def gamma2beta(gamma):
    """Computes the normalized velocity from the kinetic energy
        
    Parameters
    ----------
    gamma : array_like
        Lorentz Parameter
    Returns
    -------
    array_like
        normalized velocity
    """
    
    return np.sqrt(1-1/gamma**2)



def det_plot_scale(ps_df,cutoff = 0.75):
    """Gives the scalings for a physical particle distribution
    
    Takes the 6 dimensional phase space and returns the associated scaling.
    
    Parameters
    ----------
    ps_df : DataFrame
        Particle distribution in physical units:
        x(m),xp(rad),y(m),yp(rad),z,deltaGamma/gamma~deltaP/P
    cutoff : float
        Indicates the cutoff value for scale, e.g. 0.75 means 
        that a value of 750 will return a scale of 3 instead 
        of 0. Must be greater than 0 and equal to or less 
        than 1.
    
    Returns
    -------
    dict of {str : dict of {str : int} and {str : str}}
        A dictionary with the scaling for each dimension. 
        Can be accessed using the the coordinate name at the first 
        level ['x','xp','y','yp','z','delta'].
    """
    
    scaleSteps = 3
    maxExtents = abs(pd.concat([ps_df.max(axis=0), ps_df.min(axis=0)],axis=1))
    maxExtent = maxExtents.max(axis=1)
    maxExtent[maxExtent==0] = 1
    scale = np.floor(np.log10(maxExtent)*cutoff)
    scale = scaleSteps*np.floor(scale/scaleSteps);
    
    scale_info = {name:scale[idx]
              for idx, name in enumerate(ps_df.columns.values)}  

    return scale_info

def make_phase_space_axis_labels(dim,scale_info):
    """Makes labels for axis formatted based on scaling.
    
    Parameters
    ----------
    dim : {'x', 'xp', 'y', 'yp', 'z', 'delta'} 
        Dimension for which the label is returned.
    scale_info : {0,-3,-6,-9,-12,-15}
        Exponential Factor for scaling axis.
        
    
    Returns
    -------
    String
        Label for the specified axis, formatted 
        based on the scaling.
    """
    space_labels = { 0:'(m)',
                    -3:'(mm)',
                    -6:'(\u03BCm)',
                    -9:'(nm)',
                    -12:'(pm)',
                    -15:'(fm)'}
    transverse_angle_labels = {0:'(rad)',
                               -3:'(mrad)',
                               -6:'(\u03BCrad)',
                               -9:'(nrad)',
                               -12:'(prad)',
                               -15:'(frad)'}

    # prevent scalings greater than 0
    scale_info = np.min([0,scale_info])
    
    if(dim=='x'):
            label=f"x {space_labels[scale_info]}"
    elif(dim=='xp'):
            label=f"\u03B8<sub>x</sub> {transverse_angle_labels[scale_info]}"
    if(dim=='y'):
            label=f"y {space_labels[scale_info]}"
    elif(dim=='yp'):
            label=f"\u03B8<sub>y</sub> {transverse_angle_labels[scale_info]}"
    elif(dim=='z'):
            label=f"z {space_labels[scale_info]}"
    elif(dim=='delta'):
            label=f"U03B4 U00D7 10<sup>{int(-scale_info)}</sup>"
            
    return label

def transform_value(value):
    return 10 ** value

particle_dist = gen_dist_normal_coord_4d(gen_halton_gaussian_4d,num=10000)

controls_style = {
    # "position": "fixed",
    # "top": 0,
    # "left": 0,
    # "bottom": 0,
    #"width": "24rem",
    "padding": "2rem 6rem",
    #"background-color": "#f8f9fa",
    "fontSize": "2.0vh",
    "justifyContent": "center",
}

emittance_slider = dcc.Slider(0, 4, 0.1,
    value=1,
    id='emittance-slider',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(0,5)})  

alpha_slider = dcc.Slider(-10, 10, 0.1,
    value=0,
    id='alpha-slider',
    marks= {i: {'label': f'{i}'} for i in range(-10,11,2)}) 

beta_slider = dcc.Slider(-3, 2, 0.001,
    value=0,
    id='beta-slider',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(-3,3)})

kinetic_energy_slider = dcc.Slider(0,4,0.1,
    value=1,
    id='kinetic-energy-slider',
    marks={i: {'label': '{}'.format(10 ** i)} for i in range(0,5)})

controls = html.Div([
    html.Hr(),
    dbc.Row(dbc.Col(html.P(id='emittance-output'))),
    dbc.Row(dbc.Col(emittance_slider)),
    html.Hr(),
    dbc.Row(dbc.Col(html.P(id='alpha-output'))),
    dbc.Row(dbc.Col(alpha_slider)),
    html.Hr(),
    dbc.Row(dbc.Col(html.P(id='beta-output'))),
    dbc.Row(dbc.Col(beta_slider)),
    html.Hr(),
    dbc.Row(dbc.Col(html.P(id='kinetic-energy-output'))),
    dbc.Row(dbc.Col(kinetic_energy_slider)), 
    html.Hr(),
],
    style=controls_style)

app = Dash(__name__,
           external_stylesheets=[dbc.themes.SOLAR],
           external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" ])

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1('Exploring Twiss Parameters'), style = {'marginLeft':'7px','marginTop':'1rem','marginBottom':'1rem','textAlign':'center'})
        ]),
    dbc.Row([
        dbc.Col(html.Div(dcc.Graph(id = 'graph'),style = {'marginLeft':'15px', 'marginTop':'15px', 'marginRight':'15px'}),
                style={'textAlign':'center','display': 'flex',"justifyContent": "center"})
        ]),
    dbc.Row([
        dbc.Col(controls)
        ])
])


@app.callback(
    Output('graph', 'figure'),
    Input('emittance-slider', 'value'),
    Input('alpha-slider', 'value'),
    Input('beta-slider', 'value'),
    Input('kinetic-energy-slider', 'value'))
def update_plot(emitn,alpha,beta,kinetic_energy):
    
    temp_dist = particle_dist.copy()
    
    emitn = transform_value(emitn)
    beta = transform_value(beta)
    kinetic_energy = transform_value(kinetic_energy)
    
    gamma0 = KE2gamma(kinetic_energy*1e6)
    beta0 = gamma2beta(gamma0)
    emit = emitn/(beta0*gamma0);
        
    # get twiss ellipse
    twiss_x = Twiss(alpha=alpha,beta=beta,emit=emit*1e-9)
    x,y = ellipse_twiss(twiss_x)
    scaleX,scaleY = list(det_plot_scale(pd.DataFrame({'x':x,'xp':y}),cutoff = 0.9).values())
    
    #transform particle distribution to new parameters
    temp_dist = transform_norm_dist(temp_dist,twiss_x,twiss_x)
    
    # fig = px.scatter(
    #     x=x*10**-scaleX, 
    #     y=y*10**-scaleY,
    #     width=600, 
    #     height=600,
    #     render_mode="svg",
    #     template="solar")
    
    # fig.update_layout(xaxis={"title": make_phase_space_axis_labels('x',scaleX)},
    #                   yaxis={"title": make_phase_space_axis_labels('xp',scaleY)},
    #                   font_size=18,
    #                   transition_duration=500)
    
    trace = go.Scatter(
        x=x*10**-scaleX,
        y=y*10**-scaleY,
        # width=600, 
        # height=600,
        # template="solar"
        marker_color='red',
        )
    
    trace_dist = go.Histogram2d(
        x=temp_dist[:,0]*10**-scaleX,
        y=temp_dist[:,1]*10**-scaleY,
        colorscale="Viridis",
        nbinsx=51,
        nbinsy=51,)
    
    layout = dict(xaxis={"title": make_phase_space_axis_labels('x',scaleX)},
                  yaxis={"title": make_phase_space_axis_labels('xp',scaleY)},
                  font_size=18,
                  transition_duration=500)
    # return fig
    return go.Figure(data=[trace_dist,trace], layout=layout)

@app.callback(
    Output(component_id='emittance-output', component_property='children'),
    Input('emittance-slider', 'value' ))
def update_emittance(value):
    return f"Normalized Emittance (nm-rad): {transform_value(value):.2f}"

@app.callback(
    Output(component_id='alpha-output', component_property='children'),
    Input('alpha-slider', 'value' ))
def update_alpha(value):
    return f"\u03B1: {value:.2f}"

@app.callback(
    Output(component_id='beta-output', component_property='children'),
    Input('beta-slider', 'value' ))
def update_beta(value):
    return f"\u03B2 (m): {transform_value(value):.2f}"

@app.callback(
    Output(component_id='kinetic-energy-output', component_property='children'),
    Input('kinetic-energy-slider', 'value' ))
def update_kinetic_energy(value):
    return f"Kinetic Energy (MeV): {transform_value(value):.2f}"

if __name__ == '__main__':
    app.run_server(debug=True)