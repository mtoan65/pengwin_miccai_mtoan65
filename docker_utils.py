import docker

def get_volume_info(container_name):
    client = docker.from_env()
    
    try:
        container = client.containers.get(container_name)
        
        # Get mount information
        mounts = container.attrs['Mounts']
        
        input_volume = None
        output_volume = None
        
        for mount in mounts:
            if mount['Destination'] == '/input':
                input_volume = mount['Source']
            elif mount['Destination'] == '/output':
                output_volume = mount['Source']
        
        return input_volume, output_volume
    
    except docker.errors.NotFound:
        print(f"Container '{container_name}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None
    finally:
        client.close()

# Example usage
# container_name = "mtoan65_submission:v2"
# input_vol, output_vol = get_volume_info(container_name)