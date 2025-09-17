#!/usr/bin/env python3
"""
xDiT Client - A client for interacting with xDiT image and video generation servers.

This client supports:
- Image generation via /generate endpoint
- Video generation via /generatevideo endpoint  
- Handling both file path and base64 encoded responses
- Automatic conversion of base64 content to files
- Flexible configuration options
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests


class XDiTClient:
    """Client for xDiT image and video generation services."""
    
    def __init__(self, base_url: str = "http://localhost:6000", timeout: int = 300):
        """
        Initialize the xDiT client.
        
        Args:
            base_url: Base URL of the xDiT server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def generate_image(self, 
                      prompt: str,
                      num_inference_steps: int = 50,
                      seed: int = 42,
                      cfg: float = 7.5,
                      height: int = 1024,
                      width: int = 1024,
                      save_disk_path: Optional[str] = None,
                      output_dir: str = "./output") -> Dict[str, Any]:
        """
        Generate an image using the xDiT image generation service.
        
        Args:
            prompt: Text prompt for image generation
            num_inference_steps: Number of inference steps
            seed: Random seed for generation
            cfg: Classifier-free guidance scale
            height: Image height in pixels
            width: Image width in pixels
            save_disk_path: Server-side save path (if None, returns base64)
            output_dir: Local directory to save decoded files
            
        Returns:
            Dictionary containing generation results and file information
        """
        endpoint = f"{self.base_url}/generate"
        
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "cfg": cfg,
            "height": height,
            "width": width
        }
        
        if save_disk_path:
            payload["save_disk_path"] = save_disk_path
            
        return self._make_request(endpoint, payload, output_dir, "image")
    
    def generate_video(self,
                      prompt: str,
                      num_inference_steps: int = 50,
                      num_frames: int = 17,
                      seed: int = 42,
                      cfg: float = 7.5,
                      height: int = 1024,
                      width: int = 1024,
                      fps: int = 8,
                      save_disk_path: Optional[str] = None,
                      output_dir: str = "./output") -> Dict[str, Any]:
        """
        Generate a video using the xDiT video generation service.
        
        Args:
            prompt: Text prompt for video generation
            num_inference_steps: Number of inference steps
            num_frames: Number of video frames
            seed: Random seed for generation
            cfg: Classifier-free guidance scale
            height: Video height in pixels
            width: Video width in pixels
            fps: Frames per second
            save_disk_path: Server-side save path (if None, returns base64)
            output_dir: Local directory to save decoded files
            
        Returns:
            Dictionary containing generation results and file information
        """
        endpoint = f"{self.base_url}/generatevideo"
        
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "seed": seed,
            "cfg": cfg,
            "height": height,
            "width": width,
            "fps": fps
        }
        
        if save_disk_path:
            payload["save_disk_path"] = save_disk_path
            
        return self._make_request(endpoint, payload, output_dir, "video")
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], 
                     output_dir: str, content_type: str) -> Dict[str, Any]:
        """
        Make a request to the server and handle the response.
        
        Args:
            endpoint: API endpoint URL
            payload: Request payload
            output_dir: Local output directory
            content_type: Type of content ("image" or "video")
            
        Returns:
            Dictionary containing results and file information
        """
        print(f"Making request to {endpoint}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            start_time = time.time()
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            request_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            print(f"Server response received in {request_time:.2f}s")
            print(f"Generation time: {result.get('elapsed_time', 'N/A')}")
            print(f"Message: {result.get('message', 'N/A')}")
            
            # Handle the response based on whether it's a file path or base64 data
            if result.get('save_to_disk', False):
                # Server saved to disk, result contains file path
                print(f"File saved on server: {result['output']}")
                result['local_file'] = None
                result['file_type'] = 'server_path'
            else:
                # Server returned base64 encoded data
                print("Received base64 encoded data, decoding...")
                local_file = self._decode_and_save(
                    result['output'], 
                    output_dir, 
                    content_type,
                    result.get('format', 'png' if content_type == 'image' else 'mp4')
                )
                result['local_file'] = local_file
                result['file_type'] = 'base64_decoded'
                print(f"File saved locally: {local_file}")
            
            return result
            
        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from server")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def _decode_and_save(self, base64_data: str, output_dir: str, 
                        content_type: str, file_format: str) -> str:
        """
        Decode base64 data and save to file.
        
        Args:
            base64_data: Base64 encoded file data
            output_dir: Output directory
            content_type: Type of content ("image" or "video")
            file_format: File format (e.g., "png", "mp4")
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"generated_{content_type}_{timestamp}.{file_format}"
        file_path = os.path.join(output_dir, filename)
        
        try:
            # Decode base64 data
            file_data = base64.b64decode(base64_data)
            
            # Write to file
            with open(file_path, 'wb') as f:
                f.write(file_data)
                
            print(f"Decoded {len(file_data)} bytes to {file_path}")
            return file_path
            
        except Exception as e:
            raise Exception(f"Failed to decode and save file: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    """Main CLI interface for the xDiT client."""
    parser = argparse.ArgumentParser(
        description="xDiT Client - Generate images and videos using xDiT services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate an image
  python client.py image "a cute rabbit" --steps 30 --seed 123

  # Generate a video
  python client.py video "a cat playing with a ball" --frames 25 --fps 12

  # Use server-side saving
  python client.py image "a landscape" --server-save-path /tmp/outputs

  # Custom server URL
  python client.py --url http://192.168.1.100:6000 image "a sunset"
        """
    )
    
    parser.add_argument('--url', default='http://localhost:6000',
                       help='Base URL of the xDiT server (default: http://localhost:6000)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')
    parser.add_argument('--output-dir', default='./output',
                       help='Local output directory for decoded files (default: ./output)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Image generation command
    img_parser = subparsers.add_parser('image', help='Generate an image')
    img_parser.add_argument('prompt', help='Text prompt for image generation')
    img_parser.add_argument('--steps', type=int, default=50,
                           help='Number of inference steps (default: 50)')
    img_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    img_parser.add_argument('--cfg', type=float, default=7.5,
                           help='Classifier-free guidance scale (default: 7.5)')
    img_parser.add_argument('--height', type=int, default=1024,
                           help='Image height (default: 1024)')
    img_parser.add_argument('--width', type=int, default=1024,
                           help='Image width (default: 1024)')
    img_parser.add_argument('--server-save-path', type=str,
                           help='Server-side save path (if not provided, returns base64)')
    
    # Video generation command
    vid_parser = subparsers.add_parser('video', help='Generate a video')
    vid_parser.add_argument('prompt', help='Text prompt for video generation')
    vid_parser.add_argument('--steps', type=int, default=50,
                           help='Number of inference steps (default: 50)')
    vid_parser.add_argument('--frames', type=int, default=17,
                           help='Number of video frames (default: 17)')
    vid_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    vid_parser.add_argument('--cfg', type=float, default=7.5,
                           help='Classifier-free guidance scale (default: 7.5)')
    vid_parser.add_argument('--height', type=int, default=1024,
                           help='Video height (default: 1024)')
    vid_parser.add_argument('--width', type=int, default=1024,
                           help='Video width (default: 1024)')
    vid_parser.add_argument('--fps', type=int, default=8,
                           help='Frames per second (default: 8)')
    vid_parser.add_argument('--server-save-path', type=str,
                           help='Server-side save path (if not provided, returns base64)')
    
    # Health check command
    subparsers.add_parser('health', help='Check server health')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize client
    client = XDiTClient(base_url=args.url, timeout=args.timeout)
    
    try:
        if args.command == 'health':
            if client.health_check():
                print("✅ Server is healthy and responsive")
                return 0
            else:
                print("❌ Server is not responding")
                return 1
                
        elif args.command == 'image':
            result = client.generate_image(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                seed=args.seed,
                cfg=args.cfg,
                height=args.height,
                width=args.width,
                save_disk_path=args.server_save_path,
                output_dir=args.output_dir
            )
            
        elif args.command == 'video':
            result = client.generate_video(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                num_frames=args.frames,
                seed=args.seed,
                cfg=args.cfg,
                height=args.height,
                width=args.width,
                fps=args.fps,
                save_disk_path=args.server_save_path,
                output_dir=args.output_dir
            )
        
        if args.command in ['image', 'video']:
            print("\n" + "="*50)
            print("GENERATION COMPLETE")
            print("="*50)
            print(f"Prompt: {args.prompt}")
            print(f"Generation time: {result.get('elapsed_time', 'N/A')}")
            print(f"File type: {result.get('file_type', 'N/A')}")
            
            if result.get('local_file'):
                print(f"Local file: {result['local_file']}")
            elif result.get('output'):
                print(f"Server file: {result['output']}")
                
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())