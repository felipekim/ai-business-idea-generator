import { useState, useEffect } from 'react'

const API_BASE_URL = 'http://localhost:5000/api'

export function useApi() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const apiCall = async (endpoint, options = {}) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      setLoading(false)
      return data
    } catch (err) {
      setError(err.message)
      setLoading(false)
      throw err
    }
  }
  
  return { apiCall, loading, error }
}

export function useIdeas(filters = {}) {
  const [ideas, setIdeas] = useState([])
  const [pagination, setPagination] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  const fetchIdeas = async (newFilters = {}) => {
    setLoading(true)
    setError(null)
    
    try {
      const params = new URLSearchParams({
        page: 1,
        per_page: 20,
        ...filters,
        ...newFilters
      })
      
      const response = await fetch(`${API_BASE_URL}/ideas?${params}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      setIdeas(data.ideas || [])
      setPagination(data.pagination || {})
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }
  
  useEffect(() => {
    fetchIdeas()
  }, [])
  
  return { 
    ideas, 
    pagination, 
    loading, 
    error, 
    refetch: fetchIdeas 
  }
}

export function useStats() {
  const [stats, setStats] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/stats`)
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const data = await response.json()
        setStats(data)
        setLoading(false)
      } catch (err) {
        setError(err.message)
        setLoading(false)
      }
    }
    
    fetchStats()
  }, [])
  
  return { stats, loading, error }
}

export function useNiches() {
  const [niches, setNiches] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  useEffect(() => {
    const fetchNiches = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/ideas/niches`)
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        
        const data = await response.json()
        setNiches(data.niches || [])
        setLoading(false)
      } catch (err) {
        setError(err.message)
        setLoading(false)
      }
    }
    
    fetchNiches()
  }, [])
  
  return { niches, loading, error }
}

