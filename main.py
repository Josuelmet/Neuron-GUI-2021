from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
import sys

import numpy as np

# Custom classes
import neuron_gui
import form
from response_generator import generate_response, generate_stimulus
from response_generator import generate_noise, kalman_filter



# Initialize constants
N = 3 # Number of neurons

class FormWindow(QtWidgets.QWidget, form.Ui_Form):
    def __init__(self, parent=None):
        super(FormWindow, self).__init__(parent)
        self.setupUi(self)



class MainApp(QtWidgets.QMainWindow, neuron_gui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        
        self.form = FormWindow()
        self.form.show()
        
        self.form.pushButton.clicked.connect(self.onStartClick)
        
        self.pushButton.clicked.connect(self.onEigenvalueClick)
        
    
    def getWeights(self):
        # Gather the weight matrix data
        W = np.zeros((N, N))
        
        W[0,0] = self.connection_1_1.value()
        W[0,1] = self.connection_2_1.value()
        W[0,2] = self.connection_3_1.value()
        
        W[1,0] = self.connection_1_2.value()
        W[1,1] = self.connection_2_2.value()
        W[1,2] = self.connection_3_2.value()
        
        W[2,0] = self.connection_1_3.value()
        W[2,1] = self.connection_2_3.value()
        W[2,2] = self.connection_3_3.value()
        
        return W
        
    
    def onEigenvalueClick(self):
        # Collect the weight matrix data
        W = self.getWeights()
        # Subtract the identity matrix
        W = W - np.eye(N)
        
        eig = np.linalg.eig(W)[0]
        
        # Display the three eigenvalues
        
        string = f'{eig[0]:.2f} \n{eig[1]:.2f} \n{eig[2]:.2f}'
        self.textBrowser.setText(string)
    
        
    def onStartClick(self):
        # Collect the weight matrix data
        W = self.getWeights()
        
        # Collect some parameters
        total_time = self.form.param_total_time.value()
        timescale = self.form.param_tau.value()
        initial_response = np.array([self.form.param_r0_1.value(),
                                     self.form.param_r0_2.value(),
                                     self.form.param_r0_3.value()])
        
        dt = 0.01
        
        # Collect stimulus parameters
        stim_freq = self.form.stimulus_frequency.value()
        stim_duty = self.form.stimulus_duty.value()
        stim_val1 = self.form.stimulus_on_val.value()
        stim_val2 = self.form.stimulus_off_val.value()
        
        # Generate the square-wave stimulus desired (can just be 0)
        generate_stimulus(T=total_time, dt=dt, f=stim_freq, duty=stim_duty,
                          amplitude1=stim_val1, amplitude2=stim_val2)
        
        # Generate noise, if desired
        generate_noise(T=total_time, dt=dt,
                       enable=self.form.checkBox_noise.isChecked(),
                       mean=self.form.param_noise_mean.value(),
                       stddev=self.form.param_noise_stddev.value())
        
        
        # If linear response is desired, run linear response generation.
        if self.form.radioButton_linear.isChecked():
            generate_response(tau=timescale, weights=W, r0=initial_response,
                              T=total_time, dt=dt, linear=True)
            
        # If nonlinear response is desired, run nonlinear response generation.
        elif self.form.radioButton_nonlinear.isChecked():
            # Gather parameters
            max_rate = self.form.param_max_rate.value()
            sigma = self.form.param_sigma.value()
            steepness = self.form.param_steepness.value()
            
            use_adapt = self.form.checkBox_adaptation.isChecked()
            tau_adapt = self.form.param_adaptation_tau.value()
            str_adapt = self.form.param_adaptation_strength.value()
            
            
            generate_response(tau=timescale, weights=W, r0=initial_response,
                              T=total_time, dt=dt,
                              linear=False, max_rate=max_rate,
                              sigma=sigma, steepness=steepness,
                              use_adaptation=use_adapt,
                              tau_adaptation=tau_adapt,
                              strength_adaptation=str_adapt)
            
            
        if self.form.checkBox_kalman.isChecked():
            m_noise = self.form.kalman_noise_meas.value()
            p_noise = self.form.kalman_noise_proc.value()
            P_init  = self.form.kalman_init_p.value()
            r_init  = self.form.kalman_init_response.value()
            
            kalman_filter(T=total_time, dt=dt,
                          measurement_std_dev = m_noise,
                          process_std_dev = p_noise,
                          initial_uncertainty = P_init,
                          initial_estimate = r_init)
            





    

if __name__ == '__main__':    
    if not QtWidgets.QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    main = MainApp()
    main.show()
    
    
    '''
    TODO:
        kalman filter
            kalman filter parameters
                linear vs. non-linear ??? maybe just linear..
         
        Add noise
        
        Put noise and response on same figure, different plots?
        Find results, prepare presentation
    ''' 