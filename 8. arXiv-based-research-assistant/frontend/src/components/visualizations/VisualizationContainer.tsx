import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Tabs, 
  Tab, 
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import ForceGraph from './ForceGraph';
import ForceGraph3D from './ForceGraph3D';
import TreeGraph from './TreeGraph';
import { 
  getConceptGraph, 
  getConceptTree, 
  getConceptRadial, 
  getConceptChord, 
  getConcept3D,
  VisualizationData
} from '../../services/visualization';

const VisualizationContainer: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedConcept, setSelectedConcept] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<VisualizationData | null>(null);
  const [treeData, setTreeData] = useState<VisualizationData | null>(null);
  const [radialData, setRadialData] = useState<VisualizationData | null>(null);
  const [chordData, setChordData] = useState<VisualizationData | null>(null);
  const [graph3DData, setGraph3DData] = useState<VisualizationData | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleConceptChange = (event: SelectChangeEvent<number>) => {
    setSelectedConcept(event.target.value as number);
  };

  // Load force graph data
  useEffect(() => {
    const loadGraphData = async () => {
      if (activeTab === 0 && !graphData) {
        setLoading(true);
        setError(null);
        try {
          const data = await getConceptGraph();
          setGraphData(data);
        } catch (err) {
          setError('Failed to load graph data');
          console.error(err);
        } finally {
          setLoading(false);
        }
      }
    };
    
    loadGraphData();
  }, [activeTab, graphData]);

  // Load tree data when concept changes
  useEffect(() => {
    const loadTreeData = async () => {
      if (activeTab === 1) {
        setLoading(true);
        setError(null);
        try {
          const data = await getConceptTree(selectedConcept);
          setTreeData(data);
        } catch (err) {
          setError('Failed to load tree data');
          console.error(err);
        } finally {
          setLoading(false);
        }
      }
    };
    
    loadTreeData();
  }, [activeTab, selectedConcept]);

  // Load radial data
  useEffect(() => {
    const loadRadialData = async () => {
      if (activeTab === 2 && !radialData) {
        setLoading(true);
        setError(null);
        try {
          const data = await getConceptRadial();
          setRadialData(data);
        } catch (err) {
          setError('Failed to load radial data');
          console.error(err);
        } finally {
          setLoading(false);
        }
      }
    };
    
    loadRadialData();
  }, [activeTab, radialData]);

  // Load chord data
  useEffect(() => {
    const loadChordData = async () => {
      if (activeTab === 3 && !chordData) {
        setLoading(true);
        setError(null);
        try {
          const data = await getConceptChord();
          setChordData(data);
        } catch (err) {
          setError('Failed to load chord data');
          console.error(err);
        } finally {
          setLoading(false);
        }
      }
    };
    
    loadChordData();
  }, [activeTab, chordData]);

  // Load 3D graph data
  useEffect(() => {
    const load3DData = async () => {
      if (activeTab === 4 && !graph3DData) {
        setLoading(true);
        setError(null);
        try {
          const data = await getConcept3D();
          setGraph3DData(data);
        } catch (err) {
          setError('Failed to load 3D graph data');
          console.error(err);
        } finally {
          setLoading(false);
        }
      }
    };
    
    load3DData();
  }, [activeTab, graph3DData]);

  return (
    <Box sx={{ height: 'calc(100vh - 200px)', display: 'flex', flexDirection: 'column' }}>
      <Paper sx={{ mb: 2 }}>
        <Tabs value={activeTab} onChange={handleTabChange} centered>
          <Tab label="Force Graph" />
          <Tab label="Concept Tree" />
          <Tab label="Radial Layout" />
          <Tab label="Chord Diagram" />
          <Tab label="3D Network" />
        </Tabs>
      </Paper>

      {activeTab === 1 && (
        <Box sx={{ mb: 2 }}>
          <FormControl fullWidth>
            <InputLabel id="concept-select-label">Select Concept</InputLabel>
            <Select
              labelId="concept-select-label"
              id="concept-select"
              value={selectedConcept}
              label="Select Concept"
              onChange={handleConceptChange}
            >
              <MenuItem value={1}>Neural Network</MenuItem>
              <MenuItem value={2}>Deep Learning</MenuItem>
              <MenuItem value={3}>Transformer</MenuItem>
            </Select>
          </FormControl>
        </Box>
      )}

      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        {activeTab === 0 && (
          <ForceGraph 
            data={graphData} 
            loading={loading} 
            error={error} 
            title="Computer Science Concept Network" 
          />
        )}
        
        {activeTab === 1 && (
          <TreeGraph 
            data={treeData} 
            loading={loading} 
            error={error} 
            title={`Concept Hierarchy: ${selectedConcept === 1 ? 'Neural Network' : selectedConcept === 2 ? 'Deep Learning' : 'Transformer'}`} 
          />
        )}
        
        {activeTab === 2 && (
          <ForceGraph 
            data={radialData} 
            loading={loading} 
            error={error} 
            title="Radial Concept Map" 
          />
        )}
        
        {activeTab === 3 && (
          <ForceGraph 
            data={chordData} 
            loading={loading} 
            error={error} 
            title="Concept Domain Relationships" 
          />
        )}
        
        {activeTab === 4 && (
          <ForceGraph3D 
            data={graph3DData} 
            loading={loading} 
            error={error} 
            title="3D Concept Network" 
          />
        )}
      </Box>
    </Box>
  );
};

export default VisualizationContainer;
