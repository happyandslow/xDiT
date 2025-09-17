import os
import time
from cv2.gapi import video
import torch
import ray
import logging
import base64
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import argparse

from xfuser import (
    xFuserCogVideoXPipeline,
    xFuserConsisIDPipeline,
    xFuserLattePipeline,
    xFuserArgs,    
    utils,
)
from diffusers import (
    DiffusionPipeline, 
    HunyuanVideoPipeline, 
    HunyuanVideoTransformer3DModel
)

from xfuser.core.distributed import (
    get_runtime_state,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)
from diffusers.utils import export_to_video
# Define request model
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 50
    num_frames: Optional[int] = 17
    seed: Optional[int] = 42
    cfg: Optional[float] = 7.5
    save_disk_path: Optional[str] = None
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    fps: Optional[int] = 8

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A little girl is riding a bicycle at high speed. Focused, detailed, realistic.",
                "num_inference_steps": 50,
                "seed": 42,
                "cfg": 7.5,
                "height": 1024,
                "width": 1024
            }
        }

app = FastAPI()

@ray.remote(num_gpus=1)
class VideoGenerator:
    def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int, disable_warmup: bool = False):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
        # Set memory optimization environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        
        # Set CUDA memory fraction to leave some memory for other processes
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)
        
        self.rank = rank
        self.disable_warmup = disable_warmup
        self.setup_logger()
        
        # Clear any existing CUDA cache before initialization
        torch.cuda.empty_cache()
        
        self.initialize_model(xfuser_args)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args : xFuserArgs):
        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        # Remove use_resolution_binning if it exists to avoid compatibility issues
        if hasattr(self.input_config, "use_resolution_binning"):
            delattr(self.input_config, "use_resolution_binning")
        model_name = self.engine_config.model_config.model.split("/")[-1]
        pipeline_map = {
            "CogVideoX1.5-5B": xFuserCogVideoXPipeline,
            "CogVideoX-2b": xFuserCogVideoXPipeline,
            "ConsisID-preview": xFuserConsisIDPipeline,
            "Latte-1": xFuserLattePipeline,
            "HunyuanVideoHF": HunyuanVideoPipeline,
        }

        PipelineClass = pipeline_map.get(model_name)
        if PipelineClass is None:
            raise NotImplementedError(f"{model_name} is currently not supported!")

        
        self.logger.info(f"Initializing model {model_name} from {xfuser_args.model}")

        if model_name != "HunyuanVideoHF":
            self.pipe = PipelineClass.from_pretrained(
                pretrained_model_name_or_path=xfuser_args.model,
                engine_config=self.engine_config,
                torch_dtype=torch.float16,
            ).to("cuda")
            get_runtime_state().set_video_input_parameters(
                height=xfuser_args.height,
                width=xfuser_args.width,
                num_frames=xfuser_args.num_frames,
                batch_size=xfuser_args.batch_size,
                num_inference_steps=xfuser_args.num_inference_steps,
                split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
            ) 
        else:
            self.pipe = PipelineClass.from_pretrained(
                pretrained_model_name_or_path=xfuser_args.model,
                transformer=HunyuanVideoTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path=self.engine_config.model_config.model,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    revision="refs/pr/18",
                ),
                engine_config=self.engine_config,
                torch_dtype=torch.float16,
            ).to("cuda")
            initialize_runtime_state(self.pipe, self.engine_config)
            get_runtime_state().set_video_input_parameters(
                height=xfuser_args.height,
                width=xfuser_args.width,
                num_frames=xfuser_args.num_frames,
                num_inference_steps=xfuser_args.num_inference_steps,
                split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
            ) 
            # utils.parallelize_transformer(self.pipe)  
            
        self.pipe.vae.enable_tiling()
        # Memory-efficient warmup run with smaller dimensions
        if not self.disable_warmup:
            try:
                # Use smaller dimensions for warmup to reduce memory usage
                warmup_height = min(self.input_config.height, 256)
                warmup_width = min(self.input_config.width, 256)
                warmup_frames = min(getattr(self.input_config, 'num_frames', 17), 9)
                
                _ = self.pipe(
                    height=warmup_height,
                    width=warmup_width,
                    num_frames=warmup_frames,
                    prompt="",
                    num_inference_steps=1,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                )
                # Clear cache after warmup
                torch.cuda.empty_cache()
                self.logger.info(f"Warmup completed successfully with {warmup_height}x{warmup_width}, {warmup_frames} frames")
            except Exception as e:
                self.logger.warning(f"Warmup failed: {e}, continuing without warmup")
                # Clear cache even if warmup fails
                torch.cuda.empty_cache()
        else:
            self.logger.info("Warmup disabled to save memory")
        
        self.logger.info("Model initialization completed")

    def cleanup(self):
        """Clean up distributed environment and free memory"""
        try:
            if hasattr(self, 'pipe'):
                del self.pipe
            torch.cuda.empty_cache()
            get_runtime_state().destroy_distributed_env()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")

    def generate(self, request: GenerateRequest):
        try:
            start_time = time.time()
            output = self.pipe(
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                output_type="pil",
                generator=torch.Generator(device="cuda").manual_seed(request.seed),
                guidance_scale=request.cfg,
                max_sequence_length=getattr(self.input_config, 'max_sequence_length', 226)
            )
            elapsed_time = time.time() - start_time
            
            # Clear CUDA cache after generation to free memory
            torch.cuda.empty_cache()

            if self.is_output_rank():
                if request.save_disk_path:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"generated_video_{timestamp}.mp4"
                    file_path = os.path.join(request.save_disk_path, filename)
                    os.makedirs(request.save_disk_path, exist_ok=True)
                    
                    # Export video frames to MP4 file
                    export_to_video(output.frames[0], file_path, fps=request.fps)
                    
                    return {
                        "message": "Video generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": file_path,
                        "save_to_disk": True
                    }
                else:
                    # For video output without saving to disk, we'll save to a temporary file
                    # and then encode it as base64
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Export video frames to temporary MP4 file
                    export_to_video(output.frames[0], temp_path, fps=request.fps)
                    
                    # Read the video file and encode as base64
                    with open(temp_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        video_str = base64.b64encode(video_bytes).decode()
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
                    return {
                        "message": "Video generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": video_str,
                        "save_to_disk": False,
                        "format": "mp4"
                    }
            return None
        
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def is_output_rank(self):
        """
        Determines if this process should handle the output (e.g., save or return result).
        Compatible with both xfuser and diffusers pipelines.
        """
        try:
            # Try to use xfuser's runtime state if available
            from xfuser.core.distributed import get_runtime_state
            runtime_state = get_runtime_state()
            if runtime_state is not None and hasattr(runtime_state, "is_dp_last_group"):
                return runtime_state.is_dp_last_group()
        except Exception:
            pass

        # Fallback: use torch.distributed if initialized
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0

        # Fallback for single-process or non-distributed mode
        return self.rank == 0

class Engine:
    def __init__(self, world_size: int, xfuser_args: xFuserArgs):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()
        
        num_workers = world_size
        self.workers = [
            VideoGenerator.remote(xfuser_args, rank=rank, world_size=world_size, disable_warmup=args.disable_warmup)
            for rank in range(num_workers)
        ]
        
    async def generate(self, request: GenerateRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None) 

@app.post("/generatevideo")
async def generate_video(request: GenerateRequest):
    try:
        # Add input validation
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if request.height <= 0 or request.width <= 0:
            raise HTTPException(status_code=400, detail="Height and width must be positive")
        if request.num_inference_steps <= 0:
            raise HTTPException(status_code=400, detail="num_inference_steps must be positive")
            
        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xDiT HTTP Service')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP')
    parser.add_argument('--port', type=int, default=6000, help='Host Port')
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--world_size', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--num_frames', type=int, default=17, help='Number of frames')
    parser.add_argument('--height', type=int, default=512, help='Video height')
    parser.add_argument('--width', type=int, default=512, help='Video width')
    parser.add_argument('--pipefusion_parallel_degree', type=int, default=1, help='Degree of pipeline fusion parallelism')
    parser.add_argument('--ulysses_parallel_degree', type=int, default=1, help='Degree of Ulysses parallelism')
    parser.add_argument('--ring_degree', type=int, default=1, help='Degree of ring parallelism')
    parser.add_argument('--save_disk_path', type=str, default='output', help='Path to save generated images')
    parser.add_argument('--use_cfg_parallel', action='store_true', help='Whether to use CFG parallel')
    parser.add_argument('--disable_warmup', action='store_true', help='Disable warmup to save memory')
    args = parser.parse_args()

    xfuser_args = xFuserArgs(
        model=args.model_path,
        trust_remote_code=True,
        warmup_steps=1,
        use_parallel_vae=False,
        use_torch_compile=False,
        ulysses_degree=args.ulysses_parallel_degree,
        ring_degree = args.ring_degree,
        pipefusion_parallel_degree=args.pipefusion_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        dit_parallel_size=0,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    
    engine = Engine(
        world_size=args.world_size,
        xfuser_args=xfuser_args
    )
    
    # Start the server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)