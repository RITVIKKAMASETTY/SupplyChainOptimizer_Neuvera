import React, { useState, useEffect, useCallback } from 'react';
import { GoogleMap, InfoWindow, LoadScript, Marker, DirectionsRenderer, DirectionsService } from '@react-google-maps/api';
import { Clock, AlertTriangle, Navigation, BarChart2, Save, Plus, Edit, Trash, MapPin, Search, RefreshCw, Filter, Download, Upload, Truck, ArrowRight } from 'lucide-react';
import "../App.css";


const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

const mapContainerStyle = {
  width: '100%',
  height: '500px',
  borderRadius: '8px'
};

const center = {
  lat: 20.5937,
  lng: 78.9629 
};

const RouteOptimizer = () => {
  const [nodes, setNodes] = useState([
    { id: 'W1', type: 'warehouse', name: 'Warehouse 1', location: 'Bangalore', position: { lat: 12.9716, lng: 77.5946 } },
    { id: 'D1', type: 'distribution_center', name: 'Distribution Center 1', location: 'Chennai', position: { lat: 13.0827, lng: 80.2707 } },
    { id: 'S1', type: 'supplier', name: 'Supplier 1', location: 'Hyderabad', position: { lat: 17.3850, lng: 78.4867 } },
    { id: 'W2', type: 'warehouse', name: 'Warehouse 2', location: 'Bengaluru Outskirts', position: { lat: 13.0500, lng: 77.6200 } }
  ]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [directions, setDirections] = useState(null);
  const [routeStats, setRouteStats] = useState(null);
  const [optimizationInProgress, setOptimizationInProgress] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [alertType, setAlertType] = useState('info'); 
  const [showAlert, setShowAlert] = useState(false);
  const [trafficInfo, setTrafficInfo] = useState('Moderate');
  const [weatherInfo, setWeatherInfo] = useState('Clear');
  const [isLoaded, setIsLoaded] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [editingNode, setEditingNode] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState('map');
  const [showOptimizationModal, setShowOptimizationModal] = useState(false);
  const [typeFilter, setTypeFilter] = useState('all');
  const [optimizationMethod, setOptimizationMethod] = useState('default');
  const [activeStep, setActiveStep] = useState(1);
  const [directionsCalculated, setDirectionsCalculated] = useState(false);
  const [shouldCalculateDirections, setShouldCalculateDirections] = useState(false);

  const [isAddingNewNode, setIsAddingNewNode] = useState(false);


  const [newNodeForm, setNewNodeForm] = useState({
    id: '',
    type: 'warehouse',
    name: '',
    location: '',
    position: { lat: 0, lng: 0 }
  });


  const nodeIcons = {
    warehouse: "https://maps.google.com/mapfiles/ms/icons/blue-dot.png",
    distribution_center: "https://maps.google.com/mapfiles/ms/icons/green-dot.png",
    supplier: "https://maps.google.com/mapfiles/ms/icons/red-dot.png"
  };

  const filteredNodes = nodes.filter(node => {
    const matchesSearch =
      node.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = typeFilter === 'all' || node.type === typeFilter;
    return matchesSearch && matchesType;
  });


  const calculateOptimalRoute = () => {
    if (nodes.length < 2) {
      showAlertMessage("Need at least 2 nodes to calculate a route", "warning");
      return;
    }
    
    setOptimizationInProgress(true);
    

    let sortedNodes = [...nodes];
    if (optimizationMethod === 'default') {

      sortedNodes = sortedNodes.sort((a, b) => {
        const typeOrder = { supplier: 1, warehouse: 2, distribution_center: 3 };
        return typeOrder[a.type] - typeOrder[b.type];
      });
    } else if (optimizationMethod === 'nearest_neighbor') {

      const startNode = sortedNodes[0];
      const unvisitedNodes = sortedNodes.slice(1);
      const result = [startNode];
      let currentNode = startNode;
      
      while (unvisitedNodes.length > 0) {

        let minDistance = Infinity;
        let nearestNodeIndex = -1;
        
        for (let i = 0; i < unvisitedNodes.length; i++) {
          const distance = calculateDistance(
            currentNode.position.lat,
            currentNode.position.lng,
            unvisitedNodes[i].position.lat,
            unvisitedNodes[i].position.lng
          );
          
          if (distance < minDistance) {
            minDistance = distance;
            nearestNodeIndex = i;
          }
        }
        
        const nearestNode = unvisitedNodes[nearestNodeIndex];
        result.push(nearestNode);
        unvisitedNodes.splice(nearestNodeIndex, 1);
        currentNode = nearestNode;
      }
      
      sortedNodes = result;
    }


    setNodes(sortedNodes);
    

    setTimeout(() => {
      setOptimizationInProgress(false);
      showAlertMessage("Route optimized successfully!", "success");

      setDirections(null);
      setShouldCalculateDirections(true);
      setShowOptimizationModal(false);
      setActiveStep(1);
    }, 1500);
  };


  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371;
    const dLat = deg2rad(lat2 - lat1);
    const dLon = deg2rad(lon2 - lon1);
    const a =
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    const distance = R * c; 
    return distance;
  };

  const deg2rad = (deg) => {
    return deg * (Math.PI/180);
  };


  const directionsCallback = useCallback((response) => {
    if (response !== null && response.routes && response.routes.length > 0) {
      setDirections(response);
      

      const route = response.routes[0];
      setRouteStats({
        distance: route.legs.reduce((total, leg) => total + leg.distance.value, 0) / 1000, // km
        duration: route.legs.reduce((total, leg) => total + leg.duration.value, 0) / 60, // minutes
        waypoints: route.legs.length,
        trafficConditions: trafficInfo,
        fuelConsumption: (route.legs.reduce((total, leg) => total + leg.distance.value, 0) / 1000 * 0.08).toFixed(2), // Liters (assuming 0.08L/km)
        co2Emissions: (route.legs.reduce((total, leg) => total + leg.distance.value, 0) / 1000 * 2.3).toFixed(2) // kg (assuming 2.3kg/km)
      });

      setDirectionsCalculated(true);
      setShouldCalculateDirections(false);
    } else {
      showAlertMessage("Directions request failed", "error");
      setShouldCalculateDirections(false);
    }
  }, [trafficInfo]);

  const showAlertMessage = (message, type = 'info') => {
    setAlertMessage(message);
    setAlertType(type);
    setShowAlert(true);
    setTimeout(() => setShowAlert(false), 3000);
  };

 
  const addNode = () => {
    if (!newNodeForm.id || !newNodeForm.name || !newNodeForm.location) {
      showAlertMessage("Please fill all node details", "warning");
      return;
    }
    

    if (nodes.some(node => node.id === newNodeForm.id)) {
      showAlertMessage("Node ID already exists", "error");
      return;
    }
    

    const bengaluruPosition = { lat: 12.9716, lng: 77.5946 };
    const randomOffset = () => (Math.random() - 0.5) * 0.1; 
    const newPosition = {
      lat: bengaluruPosition.lat + randomOffset(),
      lng: bengaluruPosition.lng + randomOffset()
    };
    
    const newNode = {
      ...newNodeForm,
      position: newPosition
    };
    

    setNodes(prevNodes => [...prevNodes, newNode]);
    

    setNewNodeForm({
      id: '',
      type: 'warehouse',
      name: '',
      location: '',
      position: { lat: 0, lng: 0 }
    });
    
 
    setIsAddingNewNode(false);
    
    showAlertMessage("Node added successfully", "success");
    

    setDirections(null);
    setDirectionsCalculated(false);
    setShouldCalculateDirections(true);
  };


  const updateNode = () => {
    if (!editingNode) return;
    
    const updatedNodes = nodes.map(node =>
      node.id === editingNode.id ? editingNode : node
    );
    
    setNodes(updatedNodes);
    setEditMode(false);
    setEditingNode(null);
    showAlertMessage("Node updated successfully", "success");
    

    setDirections(null);
    setDirectionsCalculated(false);
    setShouldCalculateDirections(true);
  };


  const deleteNode = (id) => {
    const updatedNodes = nodes.filter(node => node.id !== id);
    setNodes(updatedNodes);
    
    if (selectedNode && selectedNode.id === id) {
      setSelectedNode(null);
    }
    
    showAlertMessage("Node deleted successfully", "success");
    

    setDirections(null);
    setDirectionsCalculated(false);
    setShouldCalculateDirections(true);
  };


  const startEditNode = (node) => {
    setEditingNode({...node});
    setEditMode(true);
    setIsAddingNewNode(false);
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setNewNodeForm({
      ...newNodeForm,
      [name]: value
    });
  };


  const handleEditingNodeChange = (e) => {
    const { name, value } = e.target;
    setEditingNode({
      ...editingNode,
      [name]: value
    });
  };


  const fetchWeatherData = () => {

    const weatherConditions = ['Clear', 'Rainy', 'Cloudy', 'Stormy'];
    const randomWeather = weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
    setWeatherInfo(randomWeather);
  };


  const fetchTrafficData = () => {

    const trafficConditions = ['Light', 'Moderate', 'Heavy'];
    const randomTraffic = trafficConditions[Math.floor(Math.random() * trafficConditions.length)];
    setTrafficInfo(randomTraffic);
  };


  const refreshExternalData = () => {
    fetchWeatherData();
    fetchTrafficData();
    showAlertMessage("External data refreshed", "info");
  };


  const exportData = () => {
    const dataStr = JSON.stringify(nodes, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = 'route-nodes.json';
    link.href = url;
    link.click();
    showAlertMessage("Data exported successfully", "success");
  };


  const importData = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const importedNodes = JSON.parse(e.target.result);
          if (Array.isArray(importedNodes)) {
            setNodes(importedNodes);
            showAlertMessage("Data imported successfully", "success");

            setDirections(null);
            setDirectionsCalculated(false);
            setShouldCalculateDirections(true);
          } else {
            showAlertMessage("Invalid data format", "error");
          }
        } catch (error) {
          showAlertMessage("Error parsing JSON data", "error");
        }
      };
      reader.readAsText(file);
    }
  };


  const nextStep = () => {
    setActiveStep(activeStep + 1);
  };


  const prevStep = () => {
    setActiveStep(activeStep - 1);
  };


  useEffect(() => {
    fetchWeatherData();
    fetchTrafficData();
  }, []);


  const onLoad = () => {
    setIsLoaded(true);
  };


  const getAlertClass = () => {
    switch(alertType) {
      case 'success': return 'bg-green-100 border-green-500 text-green-700';
      case 'warning': return 'bg-yellow-100 border-yellow-500 text-yellow-700';
      case 'error': return 'bg-red-100 border-red-500 text-red-700';
      default: return 'bg-blue-100 border-blue-500 text-blue-700';
    }
  };


  const OptimizationModal = () => {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-lg">
          <h2 className="text-xl font-bold mb-4">Route Optimization</h2>

          <div className="flex mb-6">
            <div className={`flex-1 text-center border-b-2 pb-2 ${activeStep === 1 ? 'border-blue-500 text-blue-500' : 'border-gray-300 text-gray-500'}`}>
              Parameters
            </div>
            <div className={`flex-1 text-center border-b-2 pb-2 ${activeStep === 2 ? 'border-blue-500 text-blue-500' : 'border-gray-300 text-gray-500'}`}>
              Constraints
            </div>
            <div className={`flex-1 text-center border-b-2 pb-2 ${activeStep === 3 ? 'border-blue-500 text-blue-500' : 'border-gray-300 text-gray-500'}`}>
              Review
            </div>
          </div>
          {activeStep === 1 && (
            <div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Optimization Method</label>
                <select
                  value={optimizationMethod}
                  onChange={(e) => setOptimizationMethod(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="default">Standard (Type-based)</option>
                  <option value="nearest_neighbor">Nearest Neighbor</option>
                  <option value="tsp">Traveling Salesman Problem</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Choose an algorithm to determine the optimal route
                </p>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Starting Point</label>
                <select
                  className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                >
                  {nodes.map(node => (
                    <option key={node.id} value={node.id}>
                      {node.name} ({node.id})
                    </option>
                  ))}
                </select>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
                <div className="flex items-center">
                  <input type="radio" id="time" name="priority" className="mr-2" defaultChecked />
                  <label htmlFor="time" className="mr-6">Time</label>
                  <input type="radio" id="distance" name="priority" className="mr-2" />
                  <label htmlFor="distance" className="mr-6">Distance</label>
                  <input type="radio" id="fuel" name="priority" className="mr-2" />
                  <label htmlFor="fuel">Fuel Efficiency</label>
                </div>
              </div>
            </div>
          )}
          {activeStep === 2 && (
            <div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Consider Traffic</label>
                <div className="flex items-center">
                  <input type="checkbox" id="traffic" className="mr-2" defaultChecked />
                  <label htmlFor="traffic">Include real-time traffic data</label>
                </div>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Vehicle Type</label>
                <select className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500">
                  <option>Small Truck (up to 3 tons)</option>
                  <option>Medium Truck (3-10 tons)</option>
                  <option>Large Truck (10+ tons)</option>
                </select>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Time Constraints</label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-600">Start Time</label>
                    <input type="time" defaultValue="09:00" className="w-full p-2 border border-gray-300 rounded" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600">End Time</label>
                    <input type="time" defaultValue="17:00" className="w-full p-2 border border-gray-300 rounded" />
                  </div>
                </div>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700">Avoid</label>
                <div className="grid grid-cols-2">
                  <div>
                    <input type="checkbox" id="tolls" className="mr-2" />
                    <label htmlFor="tolls" className="text-sm">Toll Roads</label>
                  </div>
                  <div>
                    <input type="checkbox" id="highways" className="mr-2" />
                    <label htmlFor="highways" className="text-sm">Highways</label>
                  </div>
                </div>
              </div>
            </div>
          )}
          {activeStep === 3 && (
            <div>
              <h3 className="font-medium mb-2">Optimization Summary</h3>
              <div className="bg-gray-50 p-4 rounded mb-4">
                <p className="text-sm mb-2"><span className="font-medium">Method:</span> {optimizationMethod === 'default' ? 'Standard (Type-based)' : optimizationMethod === 'nearest_neighbor' ? 'Nearest Neighbor' : 'Traveling Salesman Problem'}</p>
                <p className="text-sm mb-2"><span className="font-medium">Nodes:</span> {nodes.length}</p>
                <p className="text-sm mb-2"><span className="font-medium">Priority:</span> Time</p>
                <p className="text-sm mb-2"><span className="font-medium">Traffic Consideration:</span> Yes</p>
                <p className="text-sm"><span className="font-medium">Weather Conditions:</span> {weatherInfo}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600">
                  Ready to calculate the optimal route for {nodes.length} nodes?
                </p>
              </div>
            </div>
          )}
          <div className="flex justify-between mt-6">
            <button
              onClick={() => {
                if (activeStep === 1) {
                  setShowOptimizationModal(false);
                } else {
                  prevStep();
                }
              }}
              className="bg-gray-300 hover:bg-gray-400 text-gray-800 py-2 px-4 rounded transition-colors"
            >
              {activeStep === 1 ? 'Cancel' : 'Back'}
            </button>
            <button
              onClick={() => {
                if (activeStep === 3) {
                  calculateOptimalRoute();
                } else {
                  nextStep();
                }
              }}
              className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded transition-colors"
            >
              {activeStep === 3 ? 'Calculate Route' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-4 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-800">Route Planning & Logistics Optimization</h1>
          <div className="flex space-x-2">
            <button
              onClick={exportData}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 py-1 px-3 rounded text-sm flex items-center transition-colors"
            >
              <Download size={14} className="mr-1" />
              Export
            </button>
            <label className="bg-gray-100 hover:bg-gray-200 text-gray-700 py-1 px-3 rounded text-sm flex items-center cursor-pointer transition-colors">
              <Upload size={14} className="mr-1" />
              Import
              <input
                type="file"
                accept=".json"
                onChange={importData}
                className="hidden"
              />
            </label>
          </div>
        </div>
        {showAlert && (
          <div className={`border-l-4 p-4 mb-4 rounded shadow-sm ${getAlertClass()}`}>
            <div className="flex items-center">
              <AlertTriangle className="mr-2" size={20} />
              <p>{alertMessage}</p>
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
          <div className="bg-white p-4 rounded shadow">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-bold flex items-center">
                <AlertTriangle className="mr-2" size={16} />
                External Factors
              </h2>
              <button
                onClick={refreshExternalData}
                className="text-blue-500 hover:text-blue-700 transition-colors"
                title="Refresh external data"
              >
                <RefreshCw size={16} />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Weather</label>
                <div className={`p-2 rounded ${
                  weatherInfo === 'Clear' ? 'bg-yellow-100' :
                  weatherInfo === 'Rainy' ? 'bg-blue-100' :
                  weatherInfo === 'Cloudy' ? 'bg-gray-100' : 'bg-red-100'
                }`}>
                  <span className="font-medium">{weatherInfo}</span>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Traffic</label>
                <div className={`p-2 rounded ${
                  trafficInfo === 'Light' ? 'bg-green-100' :
                  trafficInfo === 'Moderate' ? 'bg-yellow-100' : 'bg-red-100'
                }`}>
                  <span className="font-medium">{trafficInfo}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-bold mb-4 flex items-center">
              <BarChart2 className="mr-2" size={16} />
              Route Statistics
            </h2>
            {routeStats ? (
              <div className="grid grid-cols-2 gap-y-4">
                <div>
                  <p className="text-sm text-gray-500">Distance</p>
                  <p className="font-medium">{routeStats.distance.toFixed(2)} km</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Duration</p>
                  <p className="font-medium">{routeStats.duration.toFixed(0)} min</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Stops</p>
                  <p className="font-medium">{routeStats.waypoints}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Traffic</p>
                  <p className="font-medium">{routeStats.trafficConditions}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Fuel Use</p>
                  <p className="font-medium">{routeStats.fuelConsumption} L</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">CO₂ Emissions</p>
                  <p className="font-medium">{routeStats.co2Emissions} kg</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-32 text-center">
                <p className="text-sm text-gray-500 mb-2">Calculate a route to see statistics</p>
                <button
                  onClick={() => setShowOptimizationModal(true)}
                  className="text-blue-500 hover:text-blue-700 text-sm"
                >
                  Calculate now
                </button>
              </div>
            )}
          </div>
          <div className="bg-white p-4 rounded shadow">
            <h2 className="font-bold mb-4 flex items-center">
              <Navigation className="mr-2" size={16} />
              Actions
            </h2>
            <div className="space-y-3">
              <button
                onClick={() => setShowOptimizationModal(true)}
                disabled={optimizationInProgress}
                className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
              >
                {optimizationInProgress ? (
                  <>
                    <Clock className="animate-spin mr-2" size={16} />
                    Optimizing...
                  </>
                ) : (
                  <>
                    <Navigation className="mr-2" size={16} />
                    Optimize Route
                    </>
                )}
              </button>
              <button
                onClick={() => setIsAddingNewNode(true)}
                className="w-full bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
              >
                <Plus className="mr-2" size={16} />
                Add New Node
              </button>
              <div className="flex space-x-2">
                <button
                  onClick={() => setViewMode(viewMode === 'map' ? 'list' : 'map')}
                  className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 py-2 px-4 rounded flex items-center justify-center transition-colors"
                >
                  {viewMode === 'map' ? 'View List' : 'View Map'}
                </button>
                <button
                  onClick={() => {
                    setDirections(null);
                    setDirectionsCalculated(false);
                    setShouldCalculateDirections(true);
                  }}
                  className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 py-2 px-4 rounded flex items-center justify-center transition-colors"
                >
                  <RefreshCw className="mr-2" size={16} />
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </div>
        
        {/* Main content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Map and nodes list */}
          <div className="lg:col-span-2">
            {viewMode === 'map' ? (
              <div className="bg-white p-4 rounded shadow">
                <h2 className="font-bold mb-4">Route Map</h2>
                <LoadScript googleMapsApiKey={GOOGLE_MAPS_API_KEY} onLoad={onLoad}>
                  <GoogleMap
                    mapContainerStyle={mapContainerStyle}
                    center={center}
                    zoom={5}
                  >
                    {nodes.map(node => (
                      <Marker
                        key={node.id}
                        position={node.position}
                        icon={nodeIcons[node.type]}
                        onClick={() => setSelectedNode(node)}
                      />
                    ))}
                    
                    {selectedNode && (
                      <InfoWindow
                        position={selectedNode.position}
                        onCloseClick={() => setSelectedNode(null)}
                      >
                        <div className="p-2">
                          <h3 className="font-bold">{selectedNode.name}</h3>
                          <p className="text-sm">ID: {selectedNode.id}</p>
                          <p className="text-sm">Type: {selectedNode.type.replace('_', ' ')}</p>
                          <p className="text-sm">Location: {selectedNode.location}</p>
                          <div className="mt-3 flex space-x-2">
                            <button
                              onClick={() => startEditNode(selectedNode)}
                              className="bg-blue-500 hover:bg-blue-600 text-white text-xs py-1 px-2 rounded transition-colors"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => deleteNode(selectedNode.id)}
                              className="bg-red-500 hover:bg-red-600 text-white text-xs py-1 px-2 rounded transition-colors"
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      </InfoWindow>
                    )}
                    {shouldCalculateDirections && nodes.length >= 2 && (
                      <DirectionsService
                        options={{
                          origin: nodes[0].position,
                          destination: nodes[nodes.length - 1].position,
                          waypoints: nodes.slice(1, nodes.length - 1).map(node => ({
                            location: node.position,
                            stopover: true
                          })),
                          travelMode: 'DRIVING',
                          optimizeWaypoints: true
                        }}
                        callback={directionsCallback}
                      />
                    )}
                    
                    {directions && (
                      <DirectionsRenderer
                        options={{
                          directions: directions,
                          suppressMarkers: true
                        }}
                      />
                    )}
                  </GoogleMap>
                </LoadScript>
              </div>
            ) : (
              <div className="bg-white p-4 rounded shadow">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-bold">Nodes List</h2>
                  <div className="flex space-x-2">
                    <div className="relative">
                      <input
                        type="text"
                        placeholder="Search nodes..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-8 pr-2 py-1 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                      <Search size={14} className="absolute left-2 top-2 text-gray-400" />
                    </div>
                    <select
                      value={typeFilter}
                      onChange={(e) => setTypeFilter(e.target.value)}
                      className="border border-gray-300 rounded text-sm py-1 px-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="all">All Types</option>
                      <option value="warehouse">Warehouse</option>
                      <option value="distribution_center">Distribution Center</option>
                      <option value="supplier">Supplier</option>
                    </select>
                    <button
                      onClick={() => { setSearchTerm(''); setTypeFilter('all'); }}
                      className="text-gray-500 hover:text-gray-700"
                      title="Clear filters"
                    >
                      <Filter size={16} />
                    </button>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {filteredNodes.length > 0 ? (
                        filteredNodes.map(node => (
                          <tr key={node.id} className="hover:bg-gray-50">
                            <td className="px-4 py-2 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className={`w-3 h-3 rounded-full mr-2 ${
                                  node.type === 'warehouse' ? 'bg-blue-500' :
                                  node.type === 'distribution_center' ? 'bg-green-500' : 'bg-red-500'
                                }`}></div>
                                {node.type.replace('_', ' ')}
                              </div>
                            </td>
                            <td className="px-4 py-2 whitespace-nowrap">{node.id}</td>
                            <td className="px-4 py-2 whitespace-nowrap">{node.name}</td>
                            <td className="px-4 py-2 whitespace-nowrap">{node.location}</td>
                            <td className="px-4 py-2 whitespace-nowrap text-right">
                              <button
                                onClick={() => startEditNode(node)}
                                className="text-blue-500 hover:text-blue-700 mr-2"
                                title="Edit node"
                              >
                                <Edit size={16} />
                              </button>
                              <button
                                onClick={() => deleteNode(node.id)}
                                className="text-red-500 hover:text-red-700"
                                title="Delete node"
                              >
                                <Trash size={16} />
                              </button>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan="5" className="px-4 py-4 text-center text-sm text-gray-500">
                            No nodes found matching your filters.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
          <div className="lg:col-span-1">
            {isAddingNewNode ? (
              <div className="bg-white p-4 rounded shadow">
                <h2 className="font-bold mb-4 flex items-center">
                  <Plus className="mr-2" size={16} />
                  Add New Node
                </h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Node ID</label>
                    <input
                      type="text"
                      name="id"
                      value={newNodeForm.id}
                      onChange={handleFormChange}
                      placeholder="E.g., W3, D2, S4"
                      className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Node Type</label>
                    <select
                      name="type"
                      value={newNodeForm.type}
                      onChange={handleFormChange}
                      className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="warehouse">Warehouse</option>
                      <option value="distribution_center">Distribution Center</option>
                      <option value="supplier">Supplier</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                    <input
                      type="text"
                      name="name"
                      value={newNodeForm.name}
                      onChange={handleFormChange}
                      placeholder="Node name"
                      className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                    <input
                      type="text"
                      name="location"
                      value={newNodeForm.location}
                      onChange={handleFormChange}
                      placeholder="City or area"
                      className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div className="flex space-x-4 pt-2">
                    <button
                      onClick={() => setIsAddingNewNode(false)}
                      className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 py-2 px-4 rounded transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={addNode}
                      className="flex-1 bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded transition-colors"
                    >
                      Add Node
                    </button>
                  </div>
                </div>
              </div>
            ) : editMode ? (
              <div className="bg-white p-4 rounded shadow">
                <h2 className="font-bold mb-4 flex items-center">
                  <Edit className="mr-2" size={16} />
                  Edit Node
                </h2>
                {editingNode && (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Node ID</label>
                      <input
                        type="text"
                        name="id"
                        value={editingNode.id}
                        disabled
                        className="w-full p-2 border border-gray-300 rounded bg-gray-100"
                      />
                      <p className="text-xs text-gray-500 mt-1">ID cannot be changed</p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Node Type</label>
                      <select
                        name="type"
                        value={editingNode.type}
                        onChange={handleEditingNodeChange}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="warehouse">Warehouse</option>
                        <option value="distribution_center">Distribution Center</option>
                        <option value="supplier">Supplier</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                      <input
                        type="text"
                        name="name"
                        value={editingNode.name}
                        onChange={handleEditingNodeChange}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                      <input
                        type="text"
                        name="location"
                        value={editingNode.location}
                        onChange={handleEditingNodeChange}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                    <div className="flex space-x-4 pt-2">
                      <button
                        onClick={() => {
                          setEditMode(false);
                          setEditingNode(null);
                        }}
                        className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-800 py-2 px-4 rounded transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={updateNode}
                        className="flex-1 bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded transition-colors"
                      >
                        Update Node
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white p-4 rounded shadow">
                <h2 className="font-bold mb-4 flex items-center">
                  <Truck className="mr-2" size={16} />
                  Route Details
                </h2>
                <div className="space-y-3">
                  <div>
                    <h3 className="font-medium mb-2">Route Sequence</h3>
                    <ol className="list-decimal pl-5 space-y-2">
                      {nodes.map((node, index) => (
                        <li key={node.id} className="text-sm">
                          <div className="flex items-center justify-between">
                            <div>
                              <span className="font-medium">{node.name}</span>
                              <p className="text-xs text-gray-500">{node.location}</p>
                            </div>
                            {index < nodes.length - 1 && (
                              <ArrowRight size={14} className="text-gray-400" />
                            )}
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>
                  
                  <div className="pt-2">
                    <p className="text-sm text-gray-600 mb-2">
                      {nodes.length} nodes in current route. {directions ? 'Route calculated.' : 'Route not calculated yet.'}
                    </p>
                    <div className="flex justify-center">
                      <button
                        onClick={() => setShowOptimizationModal(true)}
                        className="text-blue-500 hover:text-blue-700 text-sm flex items-center transition-colors"
                      >
                        <Navigation size={14} className="mr-1" />
                        Calculate optimal route
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      {showOptimizationModal && <OptimizationModal />}
    </div>
  );
};

export default RouteOptimizer;