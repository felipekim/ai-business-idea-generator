import { Brain, Sparkles, TrendingUp } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'

export default function Header() {
  return (
    <header className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 text-white">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Brain className="h-8 w-8" />
              <Sparkles className="h-4 w-4 absolute -top-1 -right-1 text-yellow-300" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">AI Business Ideas</h1>
              <p className="text-blue-100 text-sm">Daily AI-Powered Business Concepts</p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <div className="text-center">
              <div className="text-2xl font-bold">5</div>
              <div className="text-xs text-blue-100">Ideas Daily</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">$10K</div>
              <div className="text-xs text-blue-100">Max Cost</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">AI</div>
              <div className="text-xs text-blue-100">Validated</div>
            </div>
          </div>
          
          <Button variant="secondary" className="bg-white text-purple-600 hover:bg-gray-100">
            <TrendingUp className="h-4 w-4 mr-2" />
            Get Updates
          </Button>
        </div>
        
        <div className="mt-4 text-center">
          <p className="text-lg text-blue-100">
            Discover validated AI business ideas perfect for solo, non-technical founders
          </p>
          <p className="text-sm text-blue-200 mt-1">
            Each idea scored across 6 dimensions • Updated daily • Ready to launch
          </p>
        </div>
      </div>
    </header>
  )
}

