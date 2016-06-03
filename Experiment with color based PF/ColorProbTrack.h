struct Region {
	float u;
	float v; // u and v coordinates of the center
	float u_prime;
	float v_prime; // velocities in u and v directions 
	float w;
	float h; // width and height of the bounding box
	float w_prime;
	float h_prime; // instantaneous changes of the width and height
};

struct IterData {
	Region *sample_t;
	Region *sample_t_1;
	Region *sample_t_2;
	double *sample_weight;
	double *accum_weight;
};


