import { useState } from 'react'
import './App.css'

function App() {
  const [selectedTopic, setSelectedTopic] = useState('')
  const [code, setCode] = useState('')
  const [showImage, setShowImage] = useState(false)

  const topics = [
    '1. Digital Modulation (ASK, FSK, PSK)',
    '2. Impact of AWGN noise on signal (Plot)',
    '3. BER vs SNR in AWGN noise',
    '4. Rayleigh fading channel generation',
    '5. BER vs SNR in Rayleigh fading',
    '6. Outage probability in AWGN noise',
    '7. Outage probability in Rayleigh fading',
    '8. Impact of AWGN noise (User-defined Simulink)',
    '9. BER vs SNR in AWGN (User-defined Simulink)',
    '10. BER vs SNR in Rayleigh fading (User-defined Simulink)'
  ]

  const codes = {
    '1. Digital Modulation (ASK, FSK, PSK)': `NO CODE, THIS QUESTION NEEDS BLOCK DIAGRAM ONLY, CLICK ON VIEW GRAPH TO SEE THE OUTPUT`,
    '2. Impact of AWGN noise on signal (Plot)': `numBits = 1e6; % Number of bits
snrRange = 0:1:15; % SNR values in dB
berSimulated = zeros(size(snrRange)); % Store simulated BER
berTheoretical = zeros(size(snrRange)); % Store theoretical BER
% Generate Random Data
data = randi([0 1], numBits, 1);
% BPSK Modulation: 0 -> -1, 1 -> +1
bpskModulated = 2 * data - 1;
for i = 1:length(snrRange)
   snr = snrRange(i);
  
   % Pass the signal through an AWGN channel
   rxSignal = awgn(bpskModulated, snr);
  
   % BPSK Demodulation
   receivedBits = rxSignal > 0;
  
   % Compute BER
   berSimulated(i) = sum(receivedBits ~= data) / numBits;
end
% Plot BER vs. SNR
figure;
semilogy(snrRange, berSimulated, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
%hold on;
%semilogy(snrRange, berTheoretical, 'b-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs. SNR for BPSK in AWGN');
legend('BER');
`,
    '3. BER vs SNR in AWGN noise': `fs = 1000;
t = 0:1/fs:1;
f=10;
x=sin(2*pi*f*t);
figure;
plot(t,x);
title('Origininal Signal');
xlabel('Time(s)');
ylabel('Amplitude');
grid on;
snr=10;
y=awgn(x,snr);
figure;
plot(t,y);
title('Recieved Signal with AWGN');
xlabel('Time(s)');
ylabel('Amplitude');
grid on;
`,
    '4. Rayleigh fading channel generation': `% Parameters
s = 1; % Sigma (scale parameter)
k = 100000; % Number of samples
% Simulate Rayleigh distributed samples
h = s * (randn(k, 1) + 1i * randn(k, 1));
rayleigh_samples = abs(h);
% Define theoretical PDF
t = 0:0.01:5; % Range of values for PDF
theoretical_pdf = (t / s^2) .* exp(-t.^2 / (2 * s^2));
% Plot histogram and overlay theoretical PDF
figure;
histogram(rayleigh_samples, 'Normalization', 'pdf');
hold on;
plot(t, theoretical_pdf, 'r-', 'LineWidth', 2);
title('Rayleigh Distribution');
xlabel('Value');
ylabel('Probability Density');
legend('Simulated PDF', 'Theoretical PDF');
grid on;
`,
    '5. BER vs SNR in Rayleigh fading': `numBits = 1e6; % Number of bits
snrRange = 0:1:15; % SNR values in dB
berSimulatedAWGN = zeros(size(snrRange)); % Store simulated BER for AWGN
berSimulatedRayleigh = zeros(size(snrRange)); % Store simulated BER for Rayleigh
% Generate Random Data
data = randi([0 1], numBits, 1);
% BPSK Modulation: 0 -> -1, 1 -> +1
bpskModulated = 2 * data - 1;
% AWGN Channel Simulation
for i = 1:length(snrRange)
   snr = snrRange(i);
  
   % Pass the signal through an AWGN channel
   rxSignal = awgn(bpskModulated, snr);
  
   % BPSK Demodulation
   receivedBits = rxSignal > 0;
  
   % Compute BER
   berSimulatedAWGN(i) = sum(receivedBits ~= data) / numBits;
end
% Rayleigh Fading Channel Simulation
SNR_dB = 0:1:15; % SNR values in dB
SNR_lin = 10.^(SNR_dB/10);  % Linear SNR scale
for idx = 1:length(SNR_dB)
   % 1) Generate random bits
   bits = randi([0, 1], numBits, 1);
  
   % 2) BPSK modulation: map 0 -> -1, 1 -> +1
   s_tx = 2*bits - 1;
  
   % 3) Rayleigh fading channel coefficients
   %    (Same approach as above, each sample ~ CN(0, s^2=1^2=1))
   h_chan = (randn(numBits, 1) + 1i*randn(numBits, 1));
  
   % 4) Add AWGN noise
   %    Noise variance depends on the SNR (per dimension)
   noise = sqrt(1/(2 * SNR_lin(idx))) * (randn(numBits, 1) + 1i*randn(numBits, 1));
  
   % 5) Received signal = h_chan * s_tx + noise
   y = h_chan .* s_tx + noise;
  
   % 6) Coherent detection (assume perfect channel knowledge)
   %    Equalize by dividing out the channel
   s_rx = y ./ h_chan;
  
   % 7) Demodulate: decide bit=1 if real(s_rx) > 0, else bit=0
   bits_hat = real(s_rx) > 0;
  
   % 8) Calculate simulated BER
   berSimulatedRayleigh(idx) = mean(bits ~= bits_hat);
end
% Plot BER vs. SNR for both AWGN and Rayleigh fading channels
figure;
semilogy(snrRange, berSimulatedAWGN, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy(SNR_dB, berSimulatedRayleigh, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs. SNR for BPSK in AWGN and Rayleigh Fading');
legend('AWGN', 'Rayleigh Fading');
hold off;
`,
    '6. Outage probability in AWGN noise': `% Parameters
gamma_dB = 0:1:30;                  % Average SNR in dB
gamma = 10.^(gamma_dB / 10);        % Convert to linear scale
gamma_th_dB = 5;                    % Threshold SNR in dB
gamma_th = 10^(gamma_th_dB / 10);   % Convert to linear scale
% Outage Probability (Rayleigh fading + AWGN)
P_out = 1 - exp(-gamma_th ./ gamma);
% Plot
figure;
semilogy(gamma_dB, P_out, 'b-o', 'LineWidth', 2);
grid on;
xlabel('Average SNR (dB)');
ylabel('Outage Probability');
title('Outage Probability vs. Average SNR in AWGN with Rayleigh Fading');
legend(['Threshold = ' num2str(gamma_th_dB) ' dB']);
`,
    '7. Outage probability in Rayleigh fading': `
    %% MATLAB code to simulate outage probability in Rayleigh fading (single)
    clc; clear; close all;
%% 1. Set seed for reproducibility
rng(0);  % Ensures results are repeatable
%% 2. Define parameters
N = 1e5;             % Number of realizations (Monte Carlo)
Pt_dB = -8:2:30;      % Transmit power in dB, starting from 0 dB
Pt_lin = 10.^(Pt_dB/10);  % Convert dB to linear scale
% Now, Pt_lin starts from 10^(0/10)=1, i.e., 10^0
noisePower = 1;      % Noise power (assume unity for simplicity)
SNR = Pt_lin/noisePower;
p = length(Pt_lin);
rate = 1;            % Desired transmission rate (bits/s/Hz)
gamma_th = 2^rate - 1;  % SNR threshold for outage (from rate)
%% 3. Generate Rayleigh fading channel samples
% h ~ CN(0,1): real and imaginary parts each ~ N(0, 1/2)
% We'll generate once for all simulations, re-use across powers.
h = (randn(1,N) + 1i*randn(1,N)) / sqrt(2); 
%% 4. Theoretical Outage Probability (Rayleigh)
% For a Rayleigh channel with average SNR = Pt_lin/noisePower,
% Outage Probability = P(gamma < gamma_th) = 1 - exp(-gamma_th / avgSNR)
theoOutage = 1 - exp(-gamma_th ./ (Pt_lin / noisePower));
%% 5. Simulated Outage Probability
simOutage = zeros(1,p);  % Preallocate
for i = 1:length(Pt_lin)
   % Current transmit power
   pt = Pt_lin(i);
   % Received SNR = (Pt * |h|^2) / noisePower
   gamma = (pt * abs(h).^2) / noisePower;
   % Count how many times SNR < gamma_th => Outage
   nOutage = sum(gamma < gamma_th);
   % Simulated outage probability
   simOutage(i) = nOutage / N;
end
%% 6. Plot results
semilogy(Pt_dB, theoOutage, 'b-o', 'LineWidth', 2);
hold on;
semilogy(Pt_dB, simOutage, 'r-s', 'LineWidth', 2);
grid on;
xlabel('Transmit Power, P_t (dB)');
ylabel('Outage Probability');
title('Outage Probability vs. Transmit Power (Rayleigh Fading)');
legend('Theoretical', 'Simulated', 'Location', 'best');
set(gca, 'FontSize', 12);

%MATLAB CODE FOR OUTAGE PROBABILITY IN RAYLEIGH FADING (MULTIPLE)
clc; clear; close all;
%% 1. Set seed for reproducibility
rng(0);  % Ensures results are repeatable
%% 2. Define parameters
N = 1e5;                   % Number of Monte Carlo realizations
Pt_dB = -8:2:30;           % Transmit power in dB
Pt_lin = 10.^(Pt_dB/10);    % Convert dB to linear scale (starts at 10^(-8/10))
noisePower = 1;            % Noise power (assume unity)
rate = 1;                  % Transmission rate (bits/s/Hz)
gamma_th = 2^rate - 1;     % SNR threshold for outage
%% 3. Define antenna configurations (number of receive antennas)
numAntennas = [1, 2, 4];
%% 4. Preallocate arrays for theoretical and simulated outage probabilities
p = length(Pt_lin);
theoOutage_all = zeros(length(numAntennas), p);
simOutage_all  = zeros(length(numAntennas), p);
%% 5. Loop over each antenna configuration
for idx = 1:length(numAntennas)
   L = numAntennas(idx);  % Number of receive antennas
  
   for i = 1:p
       pt = Pt_lin(i);
       avgSNR = pt / noisePower;
      
       %% 5a. Theoretical Outage Probability for L-branch MRC
       % P_out = 1 - exp(-gamma_th/avgSNR) * sum_{k=0}^{L-1} ((gamma_th/avgSNR)^k/k!)
       theoOutage_all(idx, i) = 1 - exp(-gamma_th/avgSNR) * ...
           sum((gamma_th/avgSNR).^(0:(L-1)) ./ factorial(0:(L-1)));
      
       %% 5b. Simulated Outage Probability
       % Generate L independent Rayleigh fading channels for N realizations
       h = (randn(L, N) + 1i*randn(L, N)) / sqrt(2);
       % Combined SNR with MRC: sum the channel power gains across antennas
       gamma = pt * sum(abs(h).^2, 1) / noisePower;
       simOutage_all(idx, i) = sum(gamma < gamma_th) / N;
   end
end
%% 6. Plot all results in one graph
figure;
colors = ['b', 'r', 'g'];           % Colors for different antenna configurations
markersTheo = ['o','s','d'];         % Markers for theoretical curves
markersSim = ['+','x','*'];          % Markers for simulated curves
for idx = 1:length(numAntennas)
   % Plot theoretical outage probability (solid line with marker)
   semilogy(Pt_dB, theoOutage_all(idx,:), [colors(idx) '-' markersTheo(idx)], 'LineWidth', 2);
   hold on;
   % Plot simulated outage probability (dashed line with different marker)
   semilogy(Pt_dB, simOutage_all(idx,:), [colors(idx) '--' markersSim(idx)], 'LineWidth', 2);
end
grid on;
xlabel('Transmit Power, P_t (dB)');
ylabel('Outage Probability');
title('Outage Probability vs. Transmit Power (Multiple Antennas, MRC)');
% Create legend entries for each configuration
legendEntries = {};
for idx = 1:length(numAntennas)
   legendEntries{end+1} = ['Theoretical, L = ' num2str(numAntennas(idx))];
   legendEntries{end+1} = ['Simulated, L = ' num2str(numAntennas(idx))];
end
legend(legendEntries, 'Location', 'best');
set(gca, 'FontSize', 12);
`,
    '8. Impact of AWGN noise (User-defined Simulink)': `NO CODE, THIS QUESTION NEEDS BLOCK DIAGRAM ONLY, CLICK ON VIEW GRAPH TO SEE THE OUTPUT`,
    '9. BER vs SNR in AWGN (User-defined Simulink)': `NO CODE, THIS QUESTION NEEDS BLOCK DIAGRAM ONLY, CLICK ON VIEW GRAPH TO SEE THE OUTPUT`,
    '10. BER vs SNR in Rayleigh fading (User-defined Simulink)': `NO CODE, THIS QUESTION NEEDS BLOCK DIAGRAM ONLY, CLICK ON VIEW GRAPH TO SEE THE OUTPUT`,
  }

  const handleSelect = (topic) => {
    setSelectedTopic(topic)
    setCode(codes[topic])
    setShowImage(false)
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
  }

  const getGraphImage = (topic) => {
    const index = topics.indexOf(topic) + 1
    return `/graphs/graph${index}.png`
  }

  return (
    <div style={{ padding: '20px', backgroundColor: '#001f3f', color: '#fff', fontFamily: 'Segoe UI' }}>
      <h2 style={{ color: '#FFD700' }}>Simulink MATLAB Code Viewer</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '20px' }}>
        {topics.map((topic, index) => (
          <button
            key={index}
            onClick={() => handleSelect(topic)}
            style={{
              padding: '10px',
              cursor: 'pointer',
              backgroundColor: '#0074D9',
              color: '#fff',
              border: 'none',
              borderRadius: '4px'
            }}
          >
            {topic}
          </button>
        ))}
      </div>

      {selectedTopic && (
        <div style={{ backgroundColor: '#001930', padding: '15px', borderRadius: '10px' }}>
          <h3 style={{ color: '#7FDBFF' }}>{selectedTopic}</h3>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            rows={15}
            style={{
              width: '100%',
              fontFamily: 'monospace',
              padding: '10px',
              fontSize: '14px',
              backgroundColor: '#001f3f',
              color: '#fff',
              border: '1px solid #39CCCC'
            }}
          />
          <div style={{ marginTop: '10px' }}>
            <button onClick={handleCopy} style={{ marginRight: '10px', padding: '8px 16px', backgroundColor: '#3D9970', color: '#fff', border: 'none', borderRadius: '4px' }}>
              Copy Code
            </button>
            <button onClick={() => setShowImage(!showImage)} style={{ padding: '8px 16px', backgroundColor: '#FF851B', color: '#fff', border: 'none', borderRadius: '4px' }}>
              {showImage ? 'Hide Graph' : 'Show Graph'}
            </button>
          </div>

          {showImage && (
            <div style={{ marginTop: '20px' }}>
              <img src={getGraphImage(selectedTopic)} alt="Graph" style={{ maxWidth: '100%', borderRadius: '10px', border: '1px solid #7FDBFF' }} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
