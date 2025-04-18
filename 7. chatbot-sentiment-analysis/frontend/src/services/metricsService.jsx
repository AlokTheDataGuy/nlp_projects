import axios from 'axios';

const API_URL = 'http://localhost:8000';

/**
 * Get metrics data from the API
 * @returns {Promise<Object>} - The metrics data
 */
export const getMetrics = async () => {
  try {
    const response = await axios.get(`${API_URL}/metrics`);
    return response.data;
  } catch (error) {
    console.error('Error fetching metrics:', error);
    throw error;
  }
};
