import api from './api';

export interface Node {
  id: number;
  name: string;
  group: number;
  size: number;
  x?: number;
  y?: number;
  z?: number;
}

export interface Link {
  source: number;
  target: number;
  value: number;
  relation: string;
}

export interface VisualizationData {
  nodes: Node[];
  links: Link[];
  visualization_type: string;
}

export const getConceptGraph = async (): Promise<VisualizationData> => {
  try {
    const response = await api.get('/visualizations/concept-graph');
    return response.data;
  } catch (error) {
    console.error('Error fetching concept graph:', error);
    throw error;
  }
};

export const getConceptTree = async (conceptId: number): Promise<VisualizationData> => {
  try {
    const response = await api.get(`/visualizations/concept-tree/${conceptId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching concept tree for concept ${conceptId}:`, error);
    throw error;
  }
};

export const getConceptRadial = async (): Promise<VisualizationData> => {
  try {
    const response = await api.get('/visualizations/concept-radial');
    return response.data;
  } catch (error) {
    console.error('Error fetching concept radial visualization:', error);
    throw error;
  }
};

export const getConceptChord = async (): Promise<VisualizationData> => {
  try {
    const response = await api.get('/visualizations/concept-chord');
    return response.data;
  } catch (error) {
    console.error('Error fetching concept chord visualization:', error);
    throw error;
  }
};

export const getConcept3D = async (): Promise<VisualizationData> => {
  try {
    const response = await api.get('/visualizations/concept-3d');
    return response.data;
  } catch (error) {
    console.error('Error fetching 3D concept visualization:', error);
    throw error;
  }
};
